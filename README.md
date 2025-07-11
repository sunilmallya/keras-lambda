# Keras package for AWS Lambda
This is a reference application for Inference with Keras on AWS Lambda along with deployable package dependencies

This repo is an extension of ryfeus’s Keras Lambda [pack](https://github.com/ryfeus/lambda-packs/tree/master/Keras_tensorflow). It seemed like the repo was created for doing training on Lambda, which isn't the ideal use case, hence porting it to do Inference. 

## Instructions

- I love doing everything in Jupyter these days, hence providing a notebook to code, deploy and test using Jupyter. Take a look at the keras-lambda-notebook.ipynb. If you'd still prefer the traditional way, look below

- Create a Lambda function from the CLI by running the following commands: 

```
cd keras-lambda/src
zip -9r lambda_function.zip  * 
aws lambda update-function-code --function-name keras-lambda-app --zip-file fileb://lambda_function.zip
```

- Test the Lambda function: 
```
aws lambda invoke --invocation-type RequestResponse --function-name keras-lambda-app --region us-west-2 --log-type Tail --payload '{"url": "https://images-na.ssl-images-amazon.com/images/G/01/img15/pet-products/small-tiles/23695_pets_vertical_store_dogs_small_tile_8._CB312176604_.jpg"}' output
```

- Update the Lambda function
```
aws lambda update-function-code --function-name keras-lambda-app --zip-file fileb://lambda_function.zip  --region us-west-2
```

## Creating the package on your own

```
I didn’t find any inference specific ports of Keras Tensorflow on Lambda, hence decide to create this package

: You can reproduce this by doing the following canonical steps

wget https://github.com/ryfeus/lambda-packs/tree/master/Keras_tensorflow/Pack.zip

pip install h5py 
pip install pillow

# replace function param
sed -i -e 's/require_flatten/include_top/g' keras/applications/imagenet_utils.py

# Lets make those binaries lean
find ./ -name “*.so” | xargs strip

# Also remove those .pyc files
find . -name \*.pyc -delete

# Deploy on Lambda
