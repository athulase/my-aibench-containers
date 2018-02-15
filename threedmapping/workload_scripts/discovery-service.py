#!/usr/bin/env python3
import socket
import struct
import sys
import threading
import etcd3
import yaml
import time
import os


multicast_group = '224.3.29.71'
multicast_port = 55555
client_message = b'Discovery Service Client Knocking on The Door'
server_message = b'ACK from Discovery Service Server'
discovery_service_file = "/discovery-service.yaml"


def start_discovery_server():
    # Create the socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Bind to the server address
    sock.bind(('', multicast_port))

    # Tell the operating system to add the socket to the multicast group on all interfaces.
    group = socket.inet_aton(multicast_group)
    mreq = struct.pack('4sL', group, socket.INADDR_ANY)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

    # Receive/respond loop
    while True:
        print('\nwaiting to receive message')
        data, address = sock.recvfrom(1024)

        print('received {} bytes from {}: {}'.format(len(data), address, data))
        if data == client_message:
            print('sending acknowledgement to', address)
            sock.sendto(server_message, address)


def start_discovery_client():
    # Create the datagram socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Set a timeout so the socket does not block indefinitely when trying to receive data.
    sock.settimeout(5)

    # Set the time-to-live for messages to 1 so they do not go past the local network segment.
    ttl = struct.pack('b', 1)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, ttl)

    # Send data to the multicast group
    print('sending {!r}'.format(client_message))
    sent = sock.sendto(client_message, (multicast_group, multicast_port))

    # Look for responses from all recipients
    while True:
        print('waiting to receive')
        try:
            data, server = sock.recvfrom(1024)
            if data == server_message:
                print('received {!r} from {}'.format(data, server))
                return server
        except socket.timeout:
            print('timed out, no more responses')
            sock.close()
            return None


def start_services_purge():
    etcd = etcd3.client(host="127.0.0.1")
    while True:
        for object in etcd.get_prefix("/service"):
            ip = object[0].decode('utf-8').split(":")[0]
            host_up = True if os.system("ping -W 2 -c 1 " + ip) is 0 else False
            if not host_up:
                etcd.delete(object[1].key.decode('utf8'))
        time.sleep(5)


if __name__ == "__main__":
    runner = sys.argv[1]
    if runner == "server":
        multicast_thread = threading.Thread(target=start_discovery_server)
        multicast_thread.start()
        service_purge_thread = threading.Thread(target=start_services_purge)
        service_purge_thread.start()
    elif runner == "client":
        #server_ip = start_discovery_client()
        server_ip = "discovery-service-0.aibench-static.svc.cluster.local"
        if server_ip is not None:
             print("Found the Discovery Service Server on %s" % server_ip)
             with open(discovery_service_file, "w") as f:
                 f.write("ip: %s" % server_ip)
        else:
             print("Discovery Service Server not found!")
    elif runner == "register":
        #script.py register service_name service_host service_port
        service_name = sys.argv[2]
        service_host = sys.argv[3]
        service_port = sys.argv[4]
        with open(discovery_service_file) as f:
            yaml_data = yaml.load(f)
        server_ip = yaml_data["ip"]
        etcd = etcd3.client(host=server_ip)
        etcd.put('/service/%s' % (service_name), "%s:%s" % (service_host, service_port))
    elif runner == "unregister":
        # script.py unregister service_name
        service_name = sys.argv[2]
        with open(discovery_service_file) as f:
            yaml_data = yaml.load(f)
        server_ip = yaml_data["ip"]
        etcd = etcd3.client(host=server_ip)
        etcd.delete('/service/%s' % (service_name))
    elif runner == "query":
        #script.py query service_name
        service_name = sys.argv[2]
        with open(discovery_service_file) as f:
            yaml_data = yaml.load(f)
        server_ip = yaml_data["ip"]
        etcd = etcd3.client(host=server_ip)
        response = etcd.get('/service/%s' % (service_name))
        print(response[0].decode('utf-8'))
    else:
        raise Exception("Something is wrong!")
