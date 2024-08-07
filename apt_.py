import bisect, collections, io, os, random, tempfile

# note: there is a lot of data here.
# it would make sense to split out download and deb parsing
# prior to dwarf loading.  then a big database could be made for arbitrary reuse.

# note: DWARF includes more information than just code sourcelines,
# such as data structure information.  with more familiarity or study of DWARF,
# this could further improve.

tempfile.tempdir = os.environ.get('TMPDIR')

import keys

import apt, apt.cache, apt.debfile
import elftools.elf.elffile # pyelftools

def debfnstream(deb, fn):
    return io.BytesIO(deb._debfile.data.extractdata(fn))

def getbuildids(stream):
    try:
        elffile = elftools.elf.elffile.ELFFile(stream)
    except elftools.common.exceptions.ELFError:
        return []
    return [
        note['n_desc']
        for sect in elffile.iter_sections() 
        if isinstance(sect, elftools.elf.sections.NoteSection)
        for note in sect.iter_notes()
        if note['n_type'] == 'NT_GNU_BUILD_ID'
    ]

class Packages:
    def __init__(self, sources = (
            (
                'deb deb-src',
                'http://ftp.debian.org/debian bullseye main contrib non-free',
                keys.debian_bullseye_release,
                'all,amd64,arm64,armel,armhf,i386,mips64el,mipsel,ppc64el,s390x'
            ), (
                'deb deb-src',
                'http://debug.mirrors.debian.org/debian-debug bullseye-debug main contrib non-free',
                keys.debian_bullseye_release,
                'all,amd64,arm64,armel,armhf,i386,mips64el,mipsel,ppc64el,s390x'
            ), (
                'deb deb-src',
                'http://ftp.ubuntu.com/ubuntu focal main multiverse restricted universe',
                keys.ubuntu_archive_2018,
                'amd64,i386'
            ), (
                'deb deb-src',
                'http://ports.ubuntu.com/ubuntu-ports focal main multiverse restricted universe',
                keys.ubuntu_archive_2018,
                'arm64,armhf,ppc64el,riscv64,s390x'
            ), (
                'deb',
                'http://ddebs.ubuntu.com/ focal main multiverse restricted universe',
                keys.ubuntu_dbgsym_2016,
                'amd64,arm64,armhf,i386,ppc64el,riscv64,s390x'
            ),
    ), root_dir = os.path.abspath('apt')):
        self.root_dir = root_dir
        apt.apt_pkg.init_config()
        apt.apt_pkg.config['Dir'] = self.root_dir
        self.state_dir = os.path.join(self.root_dir, 'var', 'lib')
        os.makedirs(os.path.join(self.state_dir, 'lists', 'partial'), exist_ok = True)
        self.etc_dir = os.path.join(self.root_dir, 'etc', 'apt')
        os.makedirs(os.path.join(self.etc_dir, 'sources.list.d'), exist_ok = True)
        os.makedirs(os.path.join(self.etc_dir, 'preferences.d'), exist_ok = True)
        self.cache_dir = os.path.join(self.root_dir, 'var', 'cache')
        self.archs = set()
        self.src_sites = set()
        with open(os.path.join(self.etc_dir, 'sources.list'), 'wt') as sourceslist_file:
            apt.apt_pkg.config['Dir::Etc::SourceList'] = os.path.basename(sourceslist_file.name)
    
            for debtypes, dist, key, archs in sources:
                self.archs.update(archs.split(','))
                if type(key) is str:
                    key = keys.asc2bin(key)
                distname = dist.split(' ')[0].split('/')[-1] + '_' + dist.split(' ')[1]
                keyfilename = os.path.join(self.state_dir, distname + '.pgp')
                with open(keyfilename, 'wb') as keyfile:
                    keyfile.write(key)
                for debtype in debtypes.split(' '):
                    if debtype == 'deb-src':
                        arch = ''
                        host = dist.split('://', 1)[1].split('/')[0]
                        self.src_sites.add(host)
                    else:
                        arch = f'arch={archs} '
                    sourceslist_file.write(f'{debtype} [{arch}signed-by={keyfilename}] {dist}\n')
    
        apt.apt_pkg.config['Apt::Architectures'] = ' '.join(self.archs)
    
        apt.apt_pkg.init_system()
    
        cache = apt.cache.Cache(apt.progress.text.OpProgress(), rootdir = self.root_dir)
        cache.update(apt.progress.text.AcquireProgress())
        cache.open()
        self.cache = cache
    @property
    def packages(self):
        return self.iter_packages()
    def iter_packages(self, shuffler = random.shuffle, skip = 0):
        pkgnames = set(self.cache.keys())
        pkgnames.difference_update([
            pkgname for pkgname in pkgnames
            if pkgname + '-dbgsym' not in pkgnames
            and pkgname + '-dbg' not in pkgnames
        ])
        pkgnames_list = [
            (pkgname, arch, version.version)
            for pkgname in pkgnames
            for arch in self.archs
            if (pkgname, arch) in self.cache._cache
            for version in apt.package.Package(self.cache, self.cache._cache[pkgname, arch]).versions
        ]
        shuffler(pkgnames_list)
        yield from (Package(self, name, arch, ver) for name, arch, ver in pkgnames_list[skip:])
        for pkgname, arch, version in pkgnames_list[skip:]:
            #try:
                pkg = Package(self, pkgname, arch, ver)
            #except Exception as e:
            #    print(e)
            #    import pdb; pdb.set_trace()
            #    continue
            #else:
                yield pkg

class Package:
    def __init__(self, packages, name, arch, version):
        if (name + '-dbgsym', arch) in packages.cache._cache:
            self.dbg = apt.package.Package(packages.cache, packages.cache._cache[name + '-dbgsym', arch])
        else:
            self.dbg = apt.package.Package(packages.cache, packages.cache._cache[name + '-dbg', arch])
        self.pkg = apt.package.Package(packages.cache, packages.cache._cache[name, arch])
        self.pkgver = { ver.version: ver for ver in self.pkg.versions }[version]
        self.dbgver = { ver.version: ver for ver in self.dbg.versions }[version]

        self.name = self.dbgver.source_name
        self.version = version

        self.pkg_dir = os.path.join(packages.cache_dir, self.dbgver.source_name + '-' + self.dbgver.source_version)

        if not os.path.exists(self.pkg_dir):
            os.makedirs(self.pkg_dir, exist_ok=True)
            if self.dbgver.origins[0].site in packages.src_sites:
                self.srcver = self.dbgver
            else:
                self.srcver = self.pkgver
            #import pdb; pdb.set_trace()
            self.src_path = self.srcver.fetch_source(
                destdir = self.pkg_dir,
                unpack = True,
                progress = apt.progress.text.AcquireProgress()
            )
        # walk the src tree to list src paths
        self.srcpaths_by_file = {}
        for dirpath, dirnames, filenames in os.walk(self.src_path):
            dirpath = dirpath[len(self.pkg_dir):]
            for filename in filenames:
                if filename not in self.srcpaths_by_file:
                    self.srcpaths_by_file[filename] = []
                self.srcpaths_by_file[filename].append(os.path.join(dirpath, filename))
        self.pkg_deb = self.pkgver.fetch_binary(
            destdir = self.pkg_dir,
            progress = apt.progress.text.AcquireProgress()
        )
        self.dbg_deb = self.dbgver.fetch_binary(
            destdir = self.pkg_dir,
            progress = apt.progress.text.AcquireProgress()
        )
        print(f'Parsing {self.pkg_deb} ...')
        self.pkg_deb = apt.debfile.DebPackage(self.pkg_deb)
        self.pkg_fns = self.pkg_deb.filelist

        print(f'Parsing {self.dbg_deb} ...')
        self.dbg_deb = apt.debfile.DebPackage(self.dbg_deb)
        self.dbg_fns = self.dbg_deb.filelist

        # map dbg info to binaries
        self.dbgfn_by_pkgfn = {}
        for pkgfn in self.pkg_fns:
            renamed_fn = os.path.join('usr/lib/debug', pkgfn)
            if renamed_fn in self.dbg_fns:
                dbgfn = renamed_fn
                assert pkgfn not in self.dbgfn_by_pkgfn or self.dbgfn_by_pkgfn[pkgfn] == dbgfn
                print(f'found {pkgfn} : {dbgfn}')
                self.dbgfn_by_pkgfn[pkgfn] = dbgfn
                continue
            stream = debfnstream(self.pkg_deb, pkgfn)
            for buildid in getbuildids(stream):
                buildid = os.path.join(buildid[:2], buildid[2:])
                for dbgfn in self.dbg_fns:
                    if buildid in dbgfn:
                        assert pkgfn not in self.dbgfn_by_pkgfn or self.dbgfn_by_pkgfn[pkgfn] == dbgfn
                        print(f'found {pkgfn} : {dbgfn}')
                        self.dbgfn_by_pkgfn[pkgfn] = dbgfn
        assert len(self.dbgfn_by_pkgfn) > 0

        # map lines to offsets
        self.dwarf_by_pkgfn = {
            pkgfn : print(f'parsing {dbgfn} ...') or DWARF(dbgfn, debfnstream(self.dbg_deb, dbgfn))
            for pkgfn, dbgfn in self.dbgfn_by_pkgfn.items()
        }

        # map compilation units to source code
        self.srcpaths_by_dwarf_fn = {}
        for dwarf in self.dwarf_by_pkgfn.values():
            ### for each dwarf filename, we find where it is in the source tree, or if it isn't.
            ### if it's present multiple times, we pick the longest matching suffix.
            for fn in dwarf.filenames:
                basefn = os.path.basename(fn)
                srcpaths = self.srcpaths_by_file.get(basefn, [])
                longest = 0
                revfn = fn[::-1]
                match = None
                for srcpath in srcpaths:
                    cand = os.path.commonprefix((srcpath[::-1], revfn))
                    if len(cand) > longest:
                        match = cand
                        longest = len(match)
                #    matchlength = # find suffix match length
                #if os.path.basename(fn) in self.srcpaths_by_file:
                if match is not None:
                    match = os.path.join(self.src_path, match[::-1])
                    self.srcpaths_by_dwarf_fn[(dwarf, fn)] = match

    @property
    def binaryfns(self):
        return [*self.dwarf_by_pkgfn.keys()]

    def srclines(self, dwarf, fn, line_start, line_end):
        srcpath = self.srcpaths_by_dwarf_fn.get((dwarf, fn))
        with open(srcpath, 'rt') as srcfile:
            lines = srcfile.readlines()
        return '\n'.join(lines[range[0]:range[1]])

    # this can include inlined functions too broadly 
    def addresses_to_lines(self, binfn, addr_start, addr_end):
        dwarf = self.dwarf_by_pkgfn[binfn]
        fn_lines = dwarf.addresses_to_lines(addr_start, addr_end)
        longest = 0
        for fn, lines in fn_lines.items():
            fn_srcpath = self.srcpaths_by_dwarf_fn.get((dwarf, fn))
            if fn_srcpath is not None:
                if lines[-1] - lines[0] >= longest:
                    longest = lines[-1] - lines[0]
                    range = (lines[0], lines[-1] + 1)
                    srcpath = fn_srcpath
        with open(srcpath, 'rt') as srcfile:
            lines = srcfile.readlines()
        return '\n'.join(lines[range[0]:range[1]])

# the problem with compilation units is they assume address bounds without code written to discern what they are
#class CompilationUnit:
#    @staticmethod
#    def from_stream(stream):
#        elffile = elftools.elf.elffile.ELFFile(stream)
#        dwarfinfo = elffile.get_dwarf_info()
#        return [
#            CompilationUnit(
#            for cu in dwarfinfo.iter_CUs()
#        ]
#    def __init__(self, bindeb, binfn, srcdeb, cu):
#        self.bin_fn = binfn
#        self.cu = cu

class DWARF:
    LineNumbering = collections.namedtuple('LineNumbering', 'fn addr_start addr_end line_start line_end cu lineprog entry')
    # based on https://github.com/eliben/pyelftools/blob/master/examples/dwarf_decode_address.py 2021-01
    def __init__(self, filename, stream):
        self.elffile = elftools.elf.elffile.ELFFile(stream)
        self.dwarfinfo = self.elffile.get_dwarf_info()
        #self.address_ranges_by_funcname = {}
        self.address_line_range_pairs_by_file = {}
        self.line_numberings = []
    def parse(self):
        for cu in self.dwarfinfo.iter_CUs():
            # each CU is roughly a sourcefile
            # the sourcefile's name is in the top 'DIE' of the CU: cu.get_top_DIE().get_full_path()
            #   each CU has line-address mappings for its sourcefile and its includes
            #   as well as functions, identifier names, constant data, structures, etc
            try:
                full_srcpath = cu.get_top_DIE().get_full_path()
                if full_srcpath.startswith('./'):
                    full_srcpath = full_srcpath[2:]
                base_srcpath = os.path.basename(full_srcpath)
                self.address_line_range_pairs_by_file[full_srcpath] = []
            except:
                full_srcpath = None
                base_srcpath = None

            #for die in cu.iter_DIEs():
            #    try:
            #        if die.tag == 'DW_TAG_subprogram':
            #            lowpc = die.attributes['DW_AT_low_pc'].value

            #            # DWARF v4 in section 2.17 describes how to interpret the
            #            # DW_AT_high_pc attribute based on the class of its form.
            #            # For class 'address' it's taken as an absolute address
            #            # (similarly to DW_AT_low_pc); for class 'constant', it's
            #            # an offset from DW_AT_low_pc.
            #            highpc_attr = die.attributes['DW_AT_high_pc']
            #            highpc_attr_class = describe_form_class(highpc_attr.form)
            #            if highpc_attr_class == 'address':
            #                highpc = highpc_attr.value
            #            elif highpc_attr_class == 'constant':
            #                highpc = lowpc + highpc_attr.value
            #            else:
            #                print('Error: invalid DW_AT_high_pc class:',
            #                      highpc_attr_class)
            #                continue

            #            self.address_ranges_by_funcnames[die.attributes['DW_AT_name'].value] = (lowpc, highpc)
            #    except KeyError:
            #        continue
            # line programs show the file/lines for address ranges
            lineprog = self.dwarfinfo.line_program_for_CU(cu)
            if lineprog is None:
                continue
                #import pdb; pdb.set_trace()
            prevstate = None
            for entry in lineprog.get_entries():
                # We're interested in those entries where a new state is assigned
                if entry.state is None:
                    continue
                # Looking for a range of addresses in two consecutive states.
                if prevstate:
                    filename = lineprog['file_entry'][prevstate.file - 1].name.decode()
                    if filename == base_srcpath: # not skipping code in header files for now
                        filename = full_srcpath
                    if filename not in self.address_line_range_pairs_by_file:
                        self.address_line_range_pairs_by_file[filename] = []
                    #import pdb; pdb.set_trace()
                    self.address_line_range_pairs_by_file[filename].append(((prevstate.address, entry.state.address), (prevstate.line, entry.state.line)))
                    self.line_numberings.append(DWARF.LineNumbering(filename, prevstate.address, entry.state.address, prevstate.line, entry.state.line, cu, lineprog, entry))
                if entry.state.end_sequence:
                    # For the state with `end_sequence`, `address` means the address
                    # of the first byte after the target machine instruction
                    # sequence and other information is meaningless. We clear
                    # prevstate so that it's not used in the next iteration. Address
                    # info is used in the above comparison to see if we need to use
                    # the line information for the prevstate.
                    prevstate = None
                else:
                    prevstate = entry.state
        self.line_numberings.sort(key = lambda line_numbering: line_numbering.addr_start)
        self.line_numbering_addr_starts = [line_numbering.addr_start for line_numbering in self.line_numberings]
        self.addr_start = min((line_numbering.addr_start for line_numbering in self.line_numberings))
        self.addr_end = max((line_numbering.addr_end for line_numbering in self.line_numberings))
    @property
    def filenames(self):
        return [*self.address_line_range_pairs_by_file.keys()]

    def addresses_to_lines(self, addr_start, addr_end):
        # so really we want to map address ranges to files.
        idx = bisect.bisect_left(self.line_numbering_addr_starts, addr_start)
        filelines = {}
        while self.line_numberings[idx].addr_end <= addr_end:
            numbering = self.line_numberings[idx]
            if numbering.fn not in filelines:
                filelines[numbering.fn] = []
            filelines[numbering.fn].append(numbering.line_start)
            idx += 1
        for list in filelines.values():
            list.sort()
        return filelines

    #class CompilationUnit:
    #    def __init__(self, dwarf, cu):
                
if __name__ == '__main__':
    pkgs = Packages()
    for pkg in pkgs.packages:
        print(pkg.name, pkg.binaryfns)
        #import pdb; pdb.set_trace()
        print(pkg.binaryfns)
