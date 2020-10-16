from socket import *
import threading

reply_lock = threading.Lock()
send_lock = threading.Lock()

client_skt = socket(AF_INET, SOCK_STREAM)

input_host = input(str("Enter server address you want to connect to: "))
port = input(str("Enter the server port you want to connect to: "))
name = input(str("Enter your name: "))
    
client_skt.connect((input_host, int(port)))
    
client_skt.send(name.encode())
s_name = client_skt.recv(1024)
s_name = s_name.decode()
print("Successfully joined ", s_name,"'s chat room.\nType [bye] to exit the chat room.")

def send():
    
    while True:
        
        message = input(str("Me: "))
        if message == "[bye]":
            message = "Left chat room!"
            client_skt.send(message.encode())
            print("\n")
            break
        client_skt.send(message.encode())
        
    client_skt.close()
    
def rec():
    
    while True:
        
        reply = client_skt.recv(1024)
        reply = reply.decode()
        print(s_name, ":", reply)
        
    client_skt.close()

def Main():
    
    while True:
        reply_lock.acquire()
        send_lock.acquire()
        thread_send = threading.Thread(name = send_lock, target = send)
        thread_send.start()

        thread_receive = threading.Thread(name = reply_lock, target = rec)
        thread_receive.start()
    
Main()
