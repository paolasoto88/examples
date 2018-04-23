#export PATH=/usr/local/cuda-8.0/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

python /tfdata/mnist_distributed.py --ps_hosts=tensorflow-ps-service.default.svc.cluster.local:2222 --worker_hosts=tensorflow-wk-service0.default.svc.cluster.local:2222,tensorflow-wk-service1.default.svc.cluster.local:2222 --job_name=worker --task_index=1 --data_dir=/tfdata/data02  --train_dir=/tfdata/checkpoint02/1
