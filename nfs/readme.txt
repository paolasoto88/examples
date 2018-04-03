# After the cluster is initialized, remove the taint of the master node to install the NFS server in it. 
kubectl describe nodes user1-c7b75

# Extract the taint's name and remove it. 
kubectl taint nodes --all node-role.kubernetes.io/master-
kubectl taint nodes user1-c7b75 node-role.kubernetes.io/master:NoSchedule-

# Create the PV to provision the NFS server
kubectl apply -f examples/nfs/nfs-pv-provisioning.yaml
kubectl apply -f examples/nfs/nfs-pvc-provisioning.yaml 

# check that the PVC is bound to the PV
kubectl get pv

# nfs-common libraries must be installed in every client and server.
sudo apt-get install nfs-common

# run the NFS server
kubectl apply -f examples/nfs/nfs-server.yaml

# expose the server with a service
kubectl apply -f examples/nfs/nfs-server-svc.yaml

# check that the service point to the correct endpoints of the NFS server
kubectl describe svc nfs-server

# check that the NFS server is running
kubectl get pods

# use the NFS server IP to update nfs-pv.yaml and create the dynamic provisioning 
kubectl apply -f examples/nfs/nfs-pv.yaml

# bound the PVC to the recently created PV
kubectl apply -f examples/nfs/nfs-pvc.yaml

# check that the PVC is bound to the PV
kubectl get pv

# join more nodes to the cluster with the join command
kubeadm join <master_IP>:6443 --token 7vxo29.n3t8icmxxxzd641k --discovery-token-ca-cert-hash sha256:6302d246d3b9733df6a37ac1799405cbff690b52a14cbbac74fa3c1e6d768e18

# deploy the web server
kubectl apply -f examples/nfs/webserver.yaml

# check that the web server is deployed on other nodes
kubectl get pods -owide

# expose the web server with a service
kubectl apply -f examples/nfs/webserver-svc.yaml
