import os
import shutil

class rmFolder():
	def __init__(self,dir1,dir2):
		self.dirName0 = dir1#"D:/sergi/Escritorio/images/Nueva carpeta/image/"#"/home/pi/Desktop/images/image_detection/"
		self.dirName1 = dir2#"D:/sergi/Escritorio/images/Nueva carpeta/image_detection/"#"/home/pi/Desktop/images/image/"

		self.folders = os.listdir(self.dirName1)
		print(self.folders)
		self.n_folders = len(self.folders)
		print(self.n_folders)
		self.l_folder0 = []
		self.l_folder1 = []

	def remove(self):
		if self.n_folders >=4:
			for f in self.folders[:]:
				self.rows = f.split("-")
				self.l_folder0.append(self.rows[2]+"-"+self.rows[1]+"-"+self.rows[0])
			self.l_folder0 = sorted(self.l_folder0)

			for of in self.l_folder0[:]:
				self.rows2 = of.split("-")
				self.l_folder1.append(self.rows2[2]+"-"+self.rows2[1]+"-"+self.rows2[0])

			for i in range(0,self.n_folders-3):
				shutil.rmtree(self.dirName1+str(self.l_folder1[i]))
				shutil.rmtree(self.dirName0+str(self.l_folder1[i]))
				print("Old folder "+str(self.l_folder1[i])+" succesfully removed")