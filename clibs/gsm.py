import serial
from curses import ascii
import os,binascii

class sim():
    def __init__(self, devID, APN, MQTTHost, MQTTPort, MQTTUser, MQTTPass, MQTTTopic):
        self.phone = serial.Serial("/dev/ttyAMA0", 115200, timeout=1)
        self.DeviceID = devID
        self.APN = APN
        self.MQTTHOST = MQTTHost
        self.MQTTPort = MQTTPort
        self.MQTTUsername = MQTTUser
        self.MQTTPassword = MQTTPass
        self.MQTTClientID = self.DeviceID + binascii.b2a_hex(os.urandom(3)).decode()
        self.TCP_DIR = '"TCP","' + self.MQTTHOST + '","' + self.MQTTPort + '"'
        self.MQTTTopic = MQTTTopic
        self.MQTTProtocolName = "MQTT"
        self.MQTTLVL = 0X04
        self.MQTTFlags = 0xC2
        self.MQTTKeepAlive = 60
        self.commandSent=False



    def AT(self, at_c):
        self.phone.write(at_c.encode() + b"\r")



    def write(self, txt):
        self.phone.write(txt)



    def sleep(self):
        while True:
            self.expecteda1 = "OK" 
            self.expecteda2 = None 
            if self.commandSent == False:
                print("Putting GSM module to sleep... ")
                self.AT("AT+CFUN=0")
                self.commandSent = True
            else:
                self.answer = self.check(self.expecteda1, self.expecteda2)
                if self.answer == 1:
                    print("\n")
                    self.commandSent = False
                    break



    def SendConnectPacket(self):
        print("Starting MQTT connect packet...")
        self.packet = bytearray()
        self.packet.append(0x10)
        self.MQTTProtocolNameLength = len(self.MQTTProtocolName)
        self.MQTTClientIDLength = len(self.MQTTClientID)
        self.MQTTUsernameLength = len(self.MQTTUsername)
        self.MQTTPasswordLength = len(self.MQTTPassword)
        self.dataLength = 2 + self.MQTTProtocolNameLength + 6 + self.MQTTClientIDLength + 2 + self.MQTTUsernameLength + 2 + self.MQTTPasswordLength
        self.X = self.dataLength
        
        while True:
            self.encodeByte = self.X % 128
            self.X = int(self.X / 128)
            if (self.X > 0):
                encodeByte |= 128
            self.packet += bytes([self.encodeByte])

            if(self.X <= 0):
                break

        self.packet += bytes([self.MQTTProtocolNameLength >> 8])
        self.packet += bytes([self.MQTTProtocolNameLength & 0xff])
        self.packet += bytes(self.MQTTProtocolName,'utf-8')

        self.packet += bytes([self.MQTTLVL])
        self.packet += bytes([self.MQTTFlags])

        self.packet += bytes([self.MQTTKeepAlive >> 8])
        self.packet += bytes([self.MQTTKeepAlive & 0xff])

        self.packet += bytes([self.MQTTClientIDLength >> 8])
        self.packet += bytes([self.MQTTClientIDLength & 0xff])
        self.packet += bytes(self.MQTTClientID,'utf-8')

        self.packet += bytes([self.MQTTUsernameLength >> 8])
        self.packet += bytes([self.MQTTUsernameLength & 0xff])
        self.packet += bytes(self.MQTTUsername,'utf-8')

        self.packet += bytes([self.MQTTPasswordLength >> 8])
        self.packet += bytes([self.MQTTPasswordLength & 0xff])
        self.packet += bytes(self.MQTTPassword,'utf-8')

        self.write(self.packet + chr(26).encode())
        print("MQTT connect packet sent.")



    def SendPublishPacket(self, msg):
        print("Starting publish packet...")
        self.packet = bytearray()
        self.packet.append(0x32)
        self.message = msg
        self.TopicLength = len(self.MQTTTopic)
        self.MessageLength = len(self.message)
        self.dataLength =  2 + self.TopicLength + 2 + self.MessageLength
        self.X = self.dataLength

        while True:
            self.encodeByte = self.X % 128
            self.X = int(self.X / 128)

            if (self.X > 0):
                self.encodeByte |= 128
            self.packet += bytes([self.encodeByte])

            if(self.X <= 0):
               break

        self.packet += bytes([self.TopicLength >> 8])
        self.packet += bytes([self.TopicLength & 0xff])
        self.packet += bytes(self.MQTTTopic, 'utf-8')
    
        self.packet += bytes([self.packeId >> 8])
        self.packet += bytes([self.packeId & 0xff])

        self.packet += bytes(self.message, 'utf-8')

        self.write(self.packet + chr(26).encode())
        print("MQTT publish packet sent.")



    def sendCommand(self, command, start_string, succ_string, at_flag, sent_flag, expeca1, expeca2, mqttString):
        self.expecteda1 = expeca1
        self.expecteda2 = expeca2
        self.strtString = start_string
        self.succString = succ_string
        self.commandSent = sent_flag
        self.Command = command
        self.atflag = at_flag
        self.mqttStr = mqttString

        if self.commandSent == False:
            if self.strtString != None:
                print(self.strtString)

            if self.Command == "connect":
                self.SendConnectPacket()

            elif self.Command == "publish":
                self.SendPublishPacket(self.mqttStr)

            else:
                self.AT(self.Command)
            self.commandSent = True
                
        else:
            self.acheck = self.check(self.expecteda1, self.expecteda2)
            if self.acheck == 1:
                if self.succString != None: 
                    print(self.succString) 
                print("\n")
                self.commandSent = False
                self.atflag += 1

        return self.commandSent, self.atflag;



    def read(self, expected1, expected2):
        self.serAns = self.phone.readall()
        print(self.serAns)
        self.msg = self.serAns.decode().replace("\r","").split("\n")
        self.expec1 = expected1
        self.expec2 = expected2

        self.ans = ["No answer."]

        if len(self.msg) > 1 or self.msg[0] != "":
            if "ERROR" in self.msg:
                self.ans = ["ERROR"]

            else:
                for self.s in self.msg:
                    if self.expec1 in str(self.s):
                        if self.expec2 == None:
                            self.ans = [self.s]           
                        else:
                            self.ans = self.s.split(" ")

        return self.ans



    def check(self, expected1, expected2):
        self.expec1 = expected1
        self.expec2 = expected2
        self.answer = self.read(self.expec1, self.expec2)
        print(self.answer)
        self.comparission = 0

        if self.expec1 in self.answer[0]:
            if self.expec2 == None:
                print(self.answer[0])
                self.comparission = 1

            else:
                print(self.answer[0] + " " + self.answer[1])
                if self.answer[1] == self.expec2:
                    self.comparission = 1

                else: 
                    self.comparission = 2

        else:
            if self.answer[0] == "No answer.":
                self.comparission = 0

            elif self.answer[0] == "ERROR":
                self.comparission = 3

            else: 
                print(self.answer[0])
                print("\n")
                self.comparission = 2

        return self.comparission



    def mqttPublish(self, stringMessage, pubPackId):
        self.strMsg = stringMessage
        self.sentFlag = False
        self.ATFlag = 0
        self.packeId = pubPackId

        self.packetIdb1 = self.packeId >> 8
        self.packetIdb2 = self.packeId & 0xff

        self.pubExp = "@\x02" + (bytes(self.packetIdb1) + bytes(self.packetIdb2)).decode()
        print(self.pubExp)


        while True:

            if self.ATFlag == 0:
                self.sentFlag, self.ATFlag = self.sendCommand("ATE0&W", "Disabling Echo...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 1:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CFUN=1", "Waking up GSM Module...", None, self.ATFlag, self.sentFlag, "Call Ready", None, None)

            elif self.ATFlag == 2:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPSHUT", "Closing previous GPRS...", None, self.ATFlag, self.sentFlag, "SHUT", "OK", None)

            elif self.ATFlag == 3:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPMUX=0", "Configuring single ip connection...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 4:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPRXGET=1", "Getting data from network manually...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 5:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CGATT=1", "GPRS Service's status...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 6:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CSTT=" + self.APN, "Setting APN...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 7:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIICR", "Bring up wireless connection...", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 8:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIFSR", "Getting local ip adress...", None, self.ATFlag, self.sentFlag, ".", None, None)

            elif self.ATFlag == 9:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPSTART=" + self.TCP_DIR, "Startup connection...", None, self.ATFlag, self.sentFlag, "CONNECT", None, None)

            elif self.ATFlag == 10:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPSEND", "Preparing for send data...", None, self.ATFlag, self.sentFlag, ">", None, None)

            elif self.ATFlag == 11:
                self.sentFlag, self.ATFlag = self.sendCommand("connect", None, None, self.ATFlag, self.sentFlag, "SEND", "OK", None)

            elif self.ATFlag == 12:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPRXGET=2,1024", None, "Connected to MQTT broker.", self.ATFlag, self.sentFlag, " \x02\x00\x00", None, None)

            elif self.ATFlag == 13:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPSEND", "Preparing for send data...", None, self.ATFlag, self.sentFlag, ">", None, None)

            elif self.ATFlag == 14:
                self.sentFlag, self.ATFlag = self.sendCommand("publish", None, None, self.ATFlag, self.sentFlag, "SEND", "OK", self.strMsg)

            elif self.ATFlag == 15:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CIPRXGET=2,1024", None, "Published succesfully to MQTT broker.", self.ATFlag, self.sentFlag, self.pubExp, None, None)

            elif self.ATFlag == 16:
                self.sentFlag, self.ATFlag = self.sendCommand("AT+CFUN=0", "Putting GSM module to sleep... ", None, self.ATFlag, self.sentFlag, "OK", None, None)

            elif self.ATFlag == 17:
                break