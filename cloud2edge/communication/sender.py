from socket import socket



class Sender:
    
    def __init__(self, my_ip_address, receiver_ip_address):
        self.ip_address = my_ip_address
        self.receiver_ip_address = receiver_ip_address
    
    @call
    def send_function(self, data):
        pass
    
    def send_signal(self):
        '''
        By given status, the Manager will call the Sender to send a signal to other devices.
        If there is a match, it returns a connection enabled signal to the Mananger.
        '''
        pass
        
        