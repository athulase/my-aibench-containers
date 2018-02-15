#!/bin/bash

source ~/.bashrc

OWN_IP=`hostname -i`

# rest server start
cd /var/www/html/reporting_restserver
php -S 0.0.0.0:5000 &

#apachectl start
echo -e "\n\nServerName db-server-0.aibench-static.svc.cluster.local\n\n" >> /etc/httpd/conf/httpd.conf
/usr/sbin/httpd -D FOREGROUND &

# Locate the Discovery Service and register services
unset http_proxy
unset https_proxy
cd /
python3.6 /discovery-service.py client
sleep 1
sync
python3.6 /discovery-service.py register reportip $OWN_IP 5000
