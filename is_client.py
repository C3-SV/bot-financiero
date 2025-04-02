is_abaco_client = False
client_id = None

def set_is_abaco_client(state):
    global is_abaco_client
    is_abaco_client = state
    print(f"From is_client, {is_abaco_client}")
    
def get_is_abaco_client():
    global is_abaco_client
    return is_abaco_client

def get_is_client_string():
    base_str = "\nEN ESTE MOMENTO, TRATAS CON UN "
    if get_is_abaco_client(): 
        base_str +=  "CLIENTE DE ABACO"
    else: 
        base_str += "NO CLIENTE DE ABACO"
    base_str += "\n"
    print (base_str)
    return base_str


