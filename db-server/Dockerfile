FROM 10.105.15.44:5000/centos-base:15022018154805

###############################################################################
# System dependencies
###############################################################################
RUN yum update -y && yum install -y \
    httpd \
    php70u \
    php70u-cli \
    php70u-json \
    php70u-mysqlnd

###############################################################################
# Reporting Web App
###############################################################################
COPY workload_scripts/reporting_webapp /var/www/html/
RUN chmod 777 -R /var/www/html/export

RUN cd /var/www/html; \
    curl -sS https://getcomposer.org/installer | php ; \
    php composer.phar require slim/slim "^3.0"; \
    mkdir reporting_restserver
COPY workload_scripts/reporting_restserver /var/www/html/reporting_restserver/

###############################################################################
# Others.
###############################################################################
COPY workload_scripts/start.sh /
RUN chmod 777 /*.sh; \
    chmod 777 /*.py; \
    sync

###############################################################################
# Always change to /
###############################################################################
WORKDIR /

###############################################################################
# Make sure that everything is written to disk
###############################################################################
RUN sync

###############################################################################
# Start container services
###############################################################################
ENTRYPOINT /start.sh && /bin/bash