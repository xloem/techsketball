#!/usr/bin/env bash

# run these:
# gcloud config set account YOUR_ACCOUNT@gmail.com
# gcloud projects list
# gcloud config set project YOUR_PROJECT
# gcloud services enable tpu.googleapis.com
# gcloud beta services identity create --service tpu.googleapis.com

TPU=v2-8
ZONE=europe-west4-a
VMNAME=techsketball-vm
VERSION=v2-alpha
CMDS='
sudo apt-get update
sudo apt-get -y dist-upgrade
sudo apt autoremove -y
sudo apt-get install -y git-lfs
python3 -m pip install --upgrade pip
python3 -m pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
python3 -m pip install transformers sentencepiece
git clone https://github.com/xloem/techsketball.git
echo; echo .. Please wait for restart. ..; echo  # for new system packages
sudo shutdown -r now; exit
'
delete_vm() {
	echo; echo .. Please wait for the VM to be deleted. ..; echo
	yes | gcloud alpha compute tpus tpu-vm delete "$VMNAME" --zone "$ZONE"
}

if gcloud alpha compute tpus tpu-vm create "$VMNAME" --zone "$ZONE" --accelerator-type "$TPU" --version "$VERSION"
then
	trap delete_vm EXIT
	echo "$CMDS" | gcloud alpha compute tpus tpu-vm ssh "$VMNAME" --zone "$ZONE"
	gcloud alpha compute tpus tpu-vm ssh "$VMNAME" --zone "$ZONE"
fi
