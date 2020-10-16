from socket import *
import threading

reply_lock = threading.Lock()
send_lock = threading.Lock()

# creates all the variables needed, as well as asking for an username to be used for the chat
server_skt = socket(AF_INET, SOCK_STREAM)
server_host = gethostname()
server_ip = "" # input ip - left blank on purpose
server_port = # input port - left blank on purpose
server_skt.bind(('', server_port))
print(server_host, "(", server_ip, ")")
    
# asks for username
name = input(str("Enter your name: "))
    
# enabling the server to accept communications from clients
server_skt.listen(1)
print('The server is ready to receive')
conn_skt, addr = server_skt.accept()
        
# receive username from client
s_name = conn_skt.recv(1024)
s_name = s_name.decode()
print(s_name, "has connected to the chat room.\nType [bye] to exit the chat room.")
conn_skt.send(name.encode())


def thrd_reply():
    
    while True:
        
        # receive messages
        reply = conn_skt.recv(1024)
        reply = reply.decode()
        
        # release lock on receiving exit message "[bye]"
        if reply == "[bye]":
            reply = "User left chat room!"
            conn_skt.send(reply.encode())
            print("\n")
            reply_lock.release()
            send_lock.release()
            break
        
        print(s_name,":", reply)
    
    # close connection
    conn_skt.close()
        
def thrd_send():
    
    while True:
        
        # input message
        message = input(str("Me: "))
        
        # release lock on sending exit message "[bye]"
        if message == "[bye]":
            message = "Left chat room!"
            conn_skt.send(message.encode())
            print("\n")
            reply_lock.release()
            send_lock.release()
            break
        
        # sends encoded message
        conn_skt.send(message.encode())
    
    # close connection
    conn_skt.close()
    

def Main():
    
    # acquiring and starting threads for both receiving and sending functions
    while True:
        
        reply_lock.acquire()
        send_lock.acquire()
        print('Connected to :', addr[0], ' : ', addr[1]) 
  
        # start threads for sending and receiving messages
        t_reply = threading.Thread(name = reply_lock, target = thrd_reply)
        t_reply.start()
        
        t_send = threading.Thread(name = send_lock, target = thrd_send)
        t_send.start()
        
    server_skt.close()
    
Main()
