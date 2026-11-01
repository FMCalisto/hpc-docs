# shellcheck shell=bash
# Sourced by train.sbatch and play.sbatch when streaming. Prints where to connect.
# Isaac Lab's private-network WebRTC stream (--livestream 2) listens on fixed ports:
# 49100/TCP (signalling) and 47998/UDP (video). Each node has one GPU, so only one
# streaming job runs per node and the ports don't collide.
server_ip=$(ip -4 -o addr show ens3 | awk '{print $4}' | cut -d/ -f1)
cat <<EOF

================  Isaac Sim stream  ================
Open the Isaac Sim WebRTC Streaming Client on your laptop
(on IST's network or VPN) and connect to:

    Server:       $server_ip
    Signal port:  49100 (TCP)
    Stream port:  47998 (UDP)

The stream starts once the scene has loaded, which can take a few
minutes on the first run on a node. Stop with:  scancel $SLURM_JOB_ID
=====================================================

EOF
