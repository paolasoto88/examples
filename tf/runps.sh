python /tfdata/mnist_distributed.py --ps_hosts=tensorflow-ps-service.default.svc.cluster.local:2222 \
--worker_hosts=tensorflow-wk-service0.default.svc.cluster.local:2222,tensorflow-wk-service1.default.svc.cluster.local:2222 \
--job_name=ps \
--task_index=0 \
--data_dir=/tfdata/data02  --train_dir=/tfdata/checkpoint02/1