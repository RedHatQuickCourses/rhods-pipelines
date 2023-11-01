import kfp
import os
import time
import uuid
from flask import Flask, request, make_response
from cloudevents.http import from_http


app = Flask(__name__)

@app.route('/', methods=['POST'])
def handle_event():
    app.logger.info(request.data)
    event = from_http(request.headers, request.get_data())
    trigger = event['type']
    fileobj = event['subject']
    srcbucket = event['source']
    
    print(trigger,fileobj,srcbucket)
    kfp_api = os.environ.get('KFP_API')
    kfp_name = os.environ.get('KFP_NAME')
    if trigger == 'com.amazonaws.ObjectCreated:CompleteMultipartUpload':
        kubeflow_handler(kfp_api,kfp_name,fileobj,srcbucket)

    response = make_response({
        "fileTrigger": fileobj 
    })
    response.headers["Ce-Id"] = str(uuid.uuid4())
    response.headers["Ce-specversion"] = "0.3"
    response.headers["Ce-Source"] = "minio/trigger/kubeflowtrigger"
    response.headers["Ce-Type"] = "com.redhat.odf.trigger"
    return response

def kubeflow_handler(kfp_host:str, kfp_name:str,file_obj:str, src_bucket:str)->None:
    print('kubeflow_handler...Connecting to kubeflow API.....',kfp_host)
    client = kfp.Client(host=kfp_host)
    print('Connecting to kubeflow API.....connected')
    v1 = src_bucket.split(".")
    bucket_name = v1[2]
    pipline_id  = client.get_pipeline_id(kfp_name)
    if pipline_id is None:
        print("No pipeline found with name ",kfp_name)
    else:
        try :
            exp_obj = client.get_experiment(experiment_id="EdgeTestMonitoringExperiment")
        except:
            exp_obj = client.create_experiment("EdgeTestExperiment","monitoring experiment")
        params_dict = dict(file_obj= file_obj, src_bucket = bucket_name)
        run = client.run_pipeline(exp_obj.id,"edgetest monitor run @ "+str(time.asctime()), pipeline_id = pipline_id,params=params_dict)
        print("Pipleline Run submitted ",run.id, run.pipeline_spec.parameters)
    print('kubeflow_handler.....finish')
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)