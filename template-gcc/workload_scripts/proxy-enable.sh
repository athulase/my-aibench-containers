#!/bin/bash

http=http://proxy-chain.intel.com:911
https=http://proxy-chain.intel.com:912
active=active

if [ $active = "active" ]; then
	echo "use_proxy = on" >> ~/.wgetrc

    echo "http_proxy=$http" >> ~/.bashrc
    echo "http_proxy=$http" >> /etc/environment
    echo "http_proxy=$http" >> ~/.wgetrc
    echo "http_proxy=$http" >> ~/.curlrc


    echo "https_proxy=$https" >> ~/.bashrc
    echo "https_proxy=$https" >> /etc/environment
    echo "https_proxy=$https" >> ~/.wgetrc


    source ~/.bashrc
fi