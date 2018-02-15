#!/usr/bin/env python
import os
import paramiko

username = 'root'
password = 'passwordai'
hostnam = '/hostnames.txt'
#please do not change hostnam to hostname, as it is a paramiko hard-wired variable that you shouldn't overwrite

def deploy_key(key, server, username, password):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(server, username=username, password=password)
    client.exec_command('mkdir -p ~/.ssh/')
    client.exec_command('echo "%s" >> ~/.ssh/authorized_keys' % key)
    client.exec_command('chmod 644 ~/.ssh/authorized_keys')
    client.exec_command('chmod 700 ~/.ssh/')

key = open(os.path.expanduser('~/.ssh/id_rsa.pub')).read()
with open(hostnam, 'r') as file:
    for host in file:
        deploy_key(key, host.rstrip(), username, password)

