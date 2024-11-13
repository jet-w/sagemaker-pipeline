FROM public.ecr.aws/sagemaker/sagemaker-distribution:latest-cpu
#ECR Public Gallery: https://gallery.ecr.aws/sagemaker/sagemaker-distribution
#Dockerfile Specifications: https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor-custom-images-specifications.html

ENV POETRY_VERSION=1.8.3

WORKDIR /home/sagemaker-user

ARG NB_USER="sagemaker-user"
ARG NB_UID=1000
ARG NB_GID=100
ENV MAMBA_USER=$NB_USER

COPY etc /home/sagemaker-user/
COPY steps /home/sagemaker-user/
