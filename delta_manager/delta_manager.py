import time
import serial
import serial.tools.list_ports as port_list
# import delta_manager.camera as Camera
import delta_manager.client as Client


class DeltaManager():
    SERIAL_BAUD_RATE = 38400

    def __init__(self):
        pass

    ## Gripper functions
    def connect_gripper(self, ):
        self.gripper = None
        available_ports = self.get_all_ports()
        for port in available_ports:
            port_name = self.get_port_name(port)
            if 'COM' in port_name or 'ttyACM' in port_name:  #TODO: check if this works on linux
                if 'CH340' in self.get_description(port):
                    self.gripper = serial.Serial(
                        port_name, 
                        self.SERIAL_BAUD_RATE, 
                        timeout=None
                    )    
        
        if self.gripper:
            self.gripper.write(f"\r\n".encode("utf-8"))
            print("Gripper connected")
            time.sleep(5)
        else:
            # raise Exception("Gripper not found")
            print("Gripper not found")

    def get_all_ports(self, ):
        return [tuple(p) for p in list(port_list.comports())]

    def get_port_name(self, port):
        return port[0]
    
    def get_description(self, port):
        return port[1]
    
    def open_gripper(self, ):
        self.gripper.write(f"h".encode("utf-8"))
        # self.wait_till_done_gripper()
        print("opening gripper")

    def open_gripper_slightly(self, ):
        self.gripper.write(f"o".encode("utf-8"))
        # self.wait_till_done_gripper()
        print("opening gripper a little bit")

    def close_gripper(self, ):
        self.gripper.write(f"g".encode("utf-8"))
        self.gripper.reset_input_buffer()
        # self.wait_till_done_gripper()
        print("closing gripper")

    def close_gripper_with_feedback(self, ):
        self.gripper.write(f"g".encode("utf-8"))
        print("closing gripper")
        self.gripper.reset_input_buffer()
        result = self.wait_till_done_gripper(wait_for_failed=True)
        return result

    def rotate_gripper(self, angle):
        # on degree -90:90 (its ralative to the current angle)
        angle = int(angle)
        self.gripper.write(f"r{angle}".encode("utf-8"))
        self.wait_till_done_gripper()
        print(f"rotating gripper to {angle} degrees")

    def force_gripper(self, force):
        # int from 1 to 5000
        # self.gripper.write(f"f{int(force)}".encode("utf-8"))
        # self.wait_till_done_gripper()
        print(f"setting gripper force to {force}")
    
    def wait_till_done_gripper(self, wait_for_failed=False):
        while 1:
            result = self.gripper.readline().decode("utf-8")
            print(f"Gripper: {result}")
            if result[0:4] == "Done":
                break
            elif result[0:6] == "failed" and wait_for_failed:
                break
        return result

    ## Delta functions
    def wait_till_done_robot(self, ):
        while 1:
            result = Client.Result
            print(f"Robot: {result}")
            if result == "success":
                break
        return result

    def go_home(self, ):
        self.move(0, 0, -37)
        self.wait_till_done_robot()
        print("going home")

    def move(self, x, y, z):
        if z>0:
            z = -z
            print("PLEASE ENTER NEGATIVE NUMBERS BETWEEN -37,-65")
        Client.order("move", f"{x},{y},{z}")
        self.wait_till_done_robot()
        print(f'moving to {x}, {y}, {z}')
        
    def move_with_time(self, x, y, z, t):
        if z>0:
            z = -z
            print("PLEASE ENTER NEGATIVE NUMBERS BETWEEN -37,-65")
        Client.order("movefast", f"{x},{y},{z},{t}")
        self.wait_till_done_robot()

    def read_forward(self, ):
        Client.order("command", "forward")
        robot_current_coordinate = [Client.Result[1], Client.Result[2], Client.Result[3]]
        return robot_current_coordinate

    def delta_stop_server(self, ):
        Client.order("command", "stop")
        print('Delta is now offline...')

    def delta_rotate_gripper(self, angle):
        Client.order("rotate", f"{angle}")
        print(f'Delta is rotating gripper {angle} degrees')

    def delta_open_gripper(self, ):
        Client.order("command", "opengripper")
        print('Delta is now offline...')
        
    def delta_close_gripper(self, ):
        Client.order("command", "closegripper")
        print('Delta is now offline...')
