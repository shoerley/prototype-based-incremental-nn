#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   __       ___          ___   __    __  _____ 
#  / _\     / __\ /\  /\ /___\ / /   /__\/__   \
#  \ \     / /   / /_/ ///  /// /   /_\    / /\/
#  _\ \ _ / /___/ __  // \_/// /___//__   / /   
#  \__/(_)\____/\/ /_/ \___/ \____/\__/   \/    
# 
# 21 mars 2018
# ci2_modele.py
import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

class Modele:
	'''Représente un modèle : des prototypes et des hyperparametres'''

	def __init__(self, s_inf = -1, s_conf=-1, alpha=0.5, mask=range(0,136)):
		self.nom = 0
		
		self.sz = 7 + 2*68

		# matrice numpy de données
		self.Prototypes = np.zeros((1, self.sz))
		self.indProtos = 0
 
		self.seuil_influence = s_inf
		self.seuil_confusion = s_conf
		self.alpha = alpha
 

		self.COL_ID = 0
		self.COL_CLASSE = 1
		self.COL_FREQUENCE = 2
		self.COL_S_INF = 3
		self.COL_S_CONF = 4
		self.COL_ALPHA = 5
		self.COL_VIDEO = 6

		#self.COL_REPUTATION = 7

		self.COL_CLASSE_EX = 2
		self.deb_data = 7
 
 		self.idCourant = 0

 		#mask =  range(17, 28) + range(36, 68) + range(17*2, 28*2) + range(36*2, 68*2)
		#data = range(0, 136)
		#data2 = [x*2 for x in data]
		#data = data + data2
		self.offset_data = self.deb_data
		self.MASK_PROTO = [x+self.offset_data for x in mask]

		self.offset_ex = 3
		self.MASK_EX = [x+self.offset_ex for x in mask]

	def setMask(self, mask):
		offset_data = self.deb_data
		self.MASK_PROTO = [x+self.offset_data for x in mask]

		offset_ex = 3
		self.MASK_EX = [x+self.offset_ex for x in mask]
	def sauverModele(self, fichier, nomModele='mod'):
		'''
		Le format du modèle concatène trois colonnes au format des données standard
		1- seuil d'influence
		2- seuil de confusion
		3- alpha, qui est le coefficient de rapprochement de modifierPrototype
		'''
		#sz = len(self.Prototypes)
		#inf = np.ones((sz, 1)) * self.seuil_influence
		#conf = np.ones((sz, 1)) * self.seuil_confusion
		#alpha = np.ones((sz, 1)) * self.alpha

 
		#inf = np.hstack((self.Prototypes, inf, conf, alpha))

		self.saveToHDF5(fichier, self.Prototypes, nomDonnees=nomModele)
		
		pass

	def ajouterPrototype(self, exemple):
		''''
		L'exemple passé en parametre devient un prototype.
		INutile de faire une copie, c'est fait en amont
		'''

		nvProto = np.zeros((self.sz))
		nvProto[self.COL_ID] = self.idCourant
		nvProto[self.COL_CLASSE] = exemple[self.COL_CLASSE_EX]
		nvProto[self.COL_FREQUENCE] = 0
		nvProto[self.COL_S_INF] = self.seuil_influence
		nvProto[self.COL_S_CONF] = self.seuil_confusion
		nvProto[self.COL_ALPHA] = self.alpha
		nvProto[self.COL_VIDEO] = exemple[0]
		#nvProto[self.COL_REPUTATION] = 0

		nvProto[self.MASK_PROTO] = exemple[self.MASK_EX]
		
		self.Prototypes = np.vstack((self.Prototypes, nvProto))

		self.idCourant += 1
		#self.Prototypes = np.vstack((self.Prototypes, exemple))
		#self.frequences = np.vstack((self.frequences, [1]))
		
	def modifierPrototype_2(self, idPrototypeA, idPrototypeB):
		idPrototypeA = np.where(self.Prototypes[:, self.COL_ID] == idPrototypeA)[0]
		idPrototypeB = np.where(self.Prototypes[:, self.COL_ID] == idPrototypeB)[0]

		G = self.alpha * (self.Prototypes[idPrototypeA, self.MASK_PROTO] - self.Prototypes[idPrototypeB, self.MASK_PROTO])
		
		self.Prototypes[idPrototypeA, self.MASK_PROTO] = self.Prototypes[idPrototypeA, self.MASK_PROTO] + G
		#self.Prototypes[idPrototypeA, self.COL_FREQUENCE] += 1
		#self.Prototypes[idPrototypeB, self.COL_FREQUENCE] -= 1

	def modifierPrototype(self, idPrototype, exemple):
		idPrototype = np.where(self.Prototypes[:, self.COL_ID] == idPrototype)[0]

		self.Prototypes[idPrototype, self.MASK_PROTO] = self.Prototypes[idPrototype, self.MASK_PROTO] + self.alpha * (self.Prototypes[idPrototype, self.MASK_PROTO] - exemple[self.MASK_EX])
		
		if idPrototype == 0:
			A = self.Prototypes[idPrototype, self.MASK_PROTO]
			print A[0:2]
		#self.Prototypes[idPrototype, self.COL_FREQUENCE] += 1

	def updateFrequence(self, idPrototype, incr=1, directvalue=-1):
		idPrototype = np.where(self.Prototypes[:, self.COL_ID] == idPrototype)[0]

		if directvalue != -1:
			incr = directvalue
		self.Prototypes[idPrototype, self.COL_FREQUENCE] += incr 


	def saveToHDF5(self, fichier, Data, nomDonnees='noName'):
		'''
		Sauvegarde les données au format HDF5
		'''
		hf = h5py.File(fichier, 'w')
		hf.create_dataset(nomDonnees, data=Data)
		hf.close()

	def chargerModele(self, fichier):
		'''
		Charge un fichier au format HDF5
		'''

		hf = h5py.File(fichier, 'r')


		datasetname = [n for n in hf.keys()]

		# on considère qu'on a un seul dataset
		Data = hf.get(datasetname[0])
		Data = np.array(Data)

		self.seuil_confusion = Data[0, self.COL_S_CONF]
		self.seuil_influence = Data[0, self.COL_S_INF]
		self.alpha = Data[0, self.COL_ALPHA]

		self.Prototypes = Data

	def statsOnModele(self):

		classes, cClasses = np.unique(self.Prototypes[:, self.COL_CLASSE] ,return_counts=True)
		stats = np.vstack((classes, cClasses))

		return np.transpose(stats.astype('i'))

	def makeFrequencesAPosteriori(self, fichier, dsetname='Distribution des prototypes'):
		hf = h5py.File(fichier, 'r')

		d = hf.get(dsetname)
		d = np.array(d)

		u, n = np.unique(d, return_counts=True)

		for i in range(len(u)):
			self.updateFrequence(u[i], directvalue=n[i])

	def plotFace(self, dataLineOfFace):
		offset = 7
		nbPoints = 68

		print dataLineOfFace.shape

		X = dataLineOfFace[0 : nbPoints]
		print X.shape
		Y = dataLineOfFace[nbPoints  : ]
		print Y.shape
		plt.scatter(X, Y)

		plt.show()