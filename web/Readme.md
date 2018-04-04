# deploy the web server
kubectl apply -f examples/nfs/webserver.yaml

# check that the web server is deployed on other nodes
kubectl get pods -owide

# expose the web server with a service
kubectl apply -f examples/nfs/webserver-svc.yaml
