# Face Detection as Service

Provides access to several models through a common API.

## API
- The API accepts a **POST** request with **form-data** for processing.
- The key will be 'image' of type 'file', with value the image file and content type optional (e.x. image/jpg)

Other parameters that can be sent include:
- model : (string) name of the model selected for processing (e.x. "RFB")
- version: (optional): by default it uses the latest version (e.x. "0.0.1")
- model_params: parameters for the model (if the model accepts parameters).
- mod_image: (boolean) if True the original image is sent back with the boxes of faces printed.
- label: (boolean) only in use when mod_image is True. It prints labels next to the boxes detected.

## Models:
Initial models available include:
1. rfb
2. slim

#### 1- RFB
[Receptive Field Block Net for Accurate and Fast Object Detection](https://github.com/ruinmessi/RFBNet)

Higher precision than slim. Inspired by the structure of Receptive Fields (RFs) in human visual systems, 
this model takes the relationship between the size and eccentricity of RFs into account, 
to enhance the discriminability and robustness of features. 

On top of the SSD (Single Shot Detector) there is assembled a lightweight CNN model, constructing the RFB Net detector. 


#### 2- Slim
Network backbone simplification, slightly faster but slightly lower precision.

### Curl request examples:
Note: Update the port to 8111 if using the Dockerfile

The most simple request using the default options:
```
curl -X POST http://localhost:7000/face-detection
     -F image=@/example.jpg
```

Some extra parameters:
```
curl -X POST http://localhost:7000/face-detection
     -F model=rfb
     -F mod_image=true
     -F image=@/example.jpg
```

And with extra headers:
```
curl -X POST http://localhost:7000/face-detection
     -H 'Content-Type: multipart/form-data'
     -H 'Accept: */*'
     -H 'Accept-Encoding: gzip, deflate'
     -H 'Connection: keep-alive'
     -H 'Host: localhost:7000'
     -H 'cache-control: no-cache'
     -F model=rfb
     -F mod_image=true
     -F image=@/example.jpg
```

## References:
- [Pytorch-SSD](https://github.com/qfgaohao/pytorch-ssd)
- [Libfacedetection](https://github.com/ShiqiYu/libfacedetection/)
- [RFBnet](https://github.com/ruinmessi/RFBNet)
- [RetinaFace](https://github.com/deepinsight/insightface/blob/master/RetinaFace/README.md)
