#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   __       ___          ___   __    __  _____ 
#  / _\     / __\ /\  /\ /___\ / /   /__\/__   \
#  \ \     / /   / /_/ ///  /// /   /_\    / /\/
#  _\ \ _ / /___/ __  // \_/// /___//__   / /   
#  \__/(_)\____/\/ /_/ \___/ \____/\__/   \/    
# 
# mars 2019
# ci2_classifieur.py
import sys
sys.path.insert(0, '../')

import numpy as np 
import h5py
import time
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import  KDTree
from sklearn.utils.estimator_checks import check_estimator
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances 


np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)
TROP_DE_PROTOS = -1
NO_ERROR = 0

class ClassifieurIncremental(BaseEstimator, ClassifierMixin):
	"""
	Classifieur incrémental à base de prototypes
	"""

	def __init__(self, s_inf=-1, s_conf=-1, alpha=-1, verbose=False):
		"""
		Initialise quelques variables de classe avec les défaults ou bien 
		avec les valeurs passées en paramètre :
		- s_inf 	: (réel) seuil d'influence
		- s_conf 	: (réel) seuil de confusion
		- alpha 	: (0 < alpha <= 1) coefficient de rapprochement
		- verbose 	: (True, False) verbeux ou non
		- errorState : (Global NO_ERROR, 0 ou 1)
		- nbMaxOverProto : 
		- distanceType : ('euclidean') distance à utiliser pour calculer la distance entre deux points
		"""
		self.s_inf=s_inf
		self.s_conf=s_conf
		self.alpha=alpha
		self.verbose=verbose
		self.errorState = NO_ERROR
		self.nbMaxOverProto = 4
		self.distanceType = 'euclidean'

		#self.Dist_ = np.zeros((1, 1))
		
	''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
			APPRENTISSAGE
	'''
	def fit(self, X, y=None):
		"""
		Fonction d'apprentissage
		- X : les données, sous forme de matrice nxm, avec n le nombre d'exemples, et m le nombre de variables
		- y : les labels, un vecteur ou None
		"""

		#Initialiser des attributs par rapport à la dimension des données
		self.initAttributes(X.shape[0], X.shape[1])
		vitesseProtos = [0, 0]

		# Pour chaque ex. de Test
		for i in range(X.shape[0]) :
			# le compteur d'exemple doit être connu par la classe entière
			self.i_ = i

			try:
				# self.printInfosRun()
				# affichage, si verbose
				if self.i_ % 5000 == 0:
					self.printInfosRun()
					
					# si on a passé le premier tour
					if self.i_ > 0:
						cf = self.controlFit(50)
						if cf :
							raise ValueError('Le nombre de prototypes est trop important. Abandon ...')
					
				# c'est le premier ex. : pas de prototypes
				if self.i_ == 0:
					raise IndexError('Il n y a pas de prototypes')

				# distance des protos à l'ex. courant
				# distance euclidienne si np.linalg.norm
				D = np.linalg.norm(self.Prototypes_[:self.p_, :] - X[self.i_, :], axis=1)
				#D = self.monHamming(self.Prototypes_[:self.p_, :], X[self.i_, :])
				

				# recherche des meilleurs protos
				self.chercherPmeilleur(X[self.i_, :], D)
				self.chercherPsecond(X[self.i_, :], D, self.Pmeilleur_[self.i_, self.COL_CLASSE_])

				# différence entre la dist de Pmeilleur->Ex et Psecond->Ex
				dPsecond = abs(self.Pmeilleur_[self.i_, self.COL_DISTANCE_] - self.Psecond_[self.i_, self.COL_DISTANCE_])
				
				# conditions de classif.
				self.Conditions_[i] = [self.Pmeilleur_[self.i_, self.COL_DISTANCE_] <= self.s_inf, dPsecond > self.s_conf, self.Pmeilleur_[self.i_, self.COL_CLASSE_] == y[self.i_]] 
				#print self.Conditions_[i]
				#print self.Pmeilleur_[self.i_, self.COL_DISTANCE_] <= self.s_inf, dPsecond > self.s_conf, self.Pmeilleur_[self.i_, self.COL_CLASSE_] == y[self.i_]
				
				"""
				si Pmeilleur est proche de X, qu'il est de la classe de X et que 
				Psecond est assez loin de Pmeilleur, on approche le prototype
				"""
				if 	self.Pmeilleur_[self.i_, self.COL_DISTANCE_] <= self.s_inf and self.Pmeilleur_[self.i_, self.COL_CLASSE_] == y[self.i_] and dPsecond > self.s_conf :
					self.approcherPrototype(int(self.Pmeilleur_[self.i_, self.COL_INDICE_]), X[self.i_, :])
				else:
					raise ValueError('Conditions non remplies')

				vitesseProtos.append([self.i_, self.p_])

			except ValueError :
				#s'il y a déjà trop de prototypes, on s'arrête et on met le code en erreur
				#sinon, c'est un mécanisme normal : on crée un nouveau prototype
				if self.i_ > 0 and self.nbDepassementsProto_ >= self.nbMaxOverProto:
					self.retirerNaN()
					print ('Il y a trop de protos : ', len(self.Prototypes_), 'pour ', self.i_, 'ex.')
					
					self.errorState = TROP_DE_PROTOS
					return self.errorState
				else:
					vitesseProtos.append([self.i_, self.p_])
					self.creerPrototype(X[self.i_, :], y[self.i_]) 

			except IndexError :
				"""c'est le 1er exemple : on crée un prototype à son image. Il n'a
				ni meilleur ni second meilleur prototype"""
				vitesseProtos.append([self.i_, self.p_])
				self.creerPrototype(X[self.i_, :], y[self.i_])
				self.Psecond_[self.i_,] = [np.nan, np.nan, np.nan]
				self.Pmeilleur_[self.i_,] = [np.nan, np.nan, np.nan] 
				
			except StopIteration :
				self.retirerNaN()
				return self

		# retrait des nan et des lignes nulles
		self.retirerNaN()   
		#print vitesseProtos
		return self

	def controlFit(self, pourcent):
		"""
		Vérifie qu'on est bien en deçà d'un pourcentage de prototypes.
		Le pourcentage est relatif au nombre d'ex. d'apprentissage déjà traités.
		S'il y a trop de prototypes, retourne True ; sinon, retourne False.

		Il y a une tolérance de nbDepassementProto_ prototypes. Cela permet de 
		continuer à travailler en cas de dépassements très ponctuels
		"""
		if self.p_*100/self.i_ > pourcent:
			self.nbDepassementsProto_ += 1

			if self.nbDepassementsProto_ >= self.nbMaxOverProto:
				return True
		else:
			return False

	def creerPrototype(self, x, y):
		# le compteur des protos est distinct du compteur des ex.
		# self.p_ contient à tout moment le nombre de prototypes
		self.Prototypes_[self.p_, :] = np.copy(x)
		self.ClassesPrototypes_[self.p_] = np.copy(y)
		self.p_ += 1

	def approcherPrototype(self, indProto, x):
		"""
		Approcher un prototype d'un exemple.
		Le proto est identifié par son indice indProto, et x est une copie des données de l'exemple
		"""
		#self.Prototypes_[self.p_, :] = self.Prototypes_[self.p_, :] + self.alpha * (self.Prototypes_[self.p_, :] - x)
		
		self.Prototypes_[indProto, :] = self.Prototypes_[indProto, :] + self.alpha * (self.Prototypes_[indProto, :] - x)
		
	def initAttributes(self, X_len, X_dim):
		"""
		Initialisation de divers attributs. Les types sont choisis pour peser
		moins lourd lors d'un export
		"""
		self.X_dim_ = X_dim
		self.X_len_ = X_len


		"""
		- [unused] Reponses_ : 
		- Conditions_ : si chacune des 3 conditions a été respectée pour l'ex. i
		- [unused] Dist_Pmeilleur_ : distance de l'ex. i à son meilleur proto
		- [unused] Dist_Psecond_ : distance de l'ex. i à son 2nd meilleur proto

		"""

		self.Reponses_ = np.zeros((self.X_len_,3), dtype=np.int32)
		self.Conditions_ = np.zeros((self.X_len_, 3), dtype=np.bool_)
		self.Dist_Pmeilleur_ = np.zeros((self.X_len_, 1), dtype=np.float)
		self.Dist_Psecond_ = np.zeros((self.X_len_, 1), dtype=np.float)

		"""
		- Pmeilleur_ : représente les meilleurs prototypes.
		Pmeilleur [i, :] contient des infos sur le meilleur prototype de X[i]
		Pmeilleur_[i, COL_INDICE] : indice dans Prototypes de Pmeilleur
		Pmeilleur [i, COL_CLASSE] : indice dans ClassesPrototypes_ de la classe de Pmeilleur
		Pmeilleur [i, COL_DISTANCE] : indice dans D de la distance entre Pmeilleur et X[i]

		Idem pour Psecond

		"""
		self.Pmeilleur_ = np.zeros((self.X_len_, 3))
		self.Psecond_ = np.zeros((self.X_len_, 3))

		"""
		- Prototypes_ : les données des prototypes. Un prototype est dans le même espace que 
		X : même dimension. 
		- ClassesPrototypes_ : la classe de chaque prototype (même indice)
		"""
		self.Prototypes_ = np.zeros((self.X_len_, self.X_dim_))
		self.ClassesPrototypes_ = np.zeros((self.X_len_, 1))

		self.nbDepassementsProto_ = 0

		self.initCols()
 
	def initCols(self):
		"""
		Numéros de colonnes pour indexer Pmeilleur et Psecond
		Initialisation du compteur de proto
		"""
		self.COL_INDICE_ = 0
		self.COL_CLASSE_ = 1
		self.COL_DISTANCE_ = 2
		self.p_= 0

	def chercherPmeilleur(self, x, D):
		# Pmeilleur est simplement le plus proche de l'ex.
		ind = np.argmin(D) 

		self.Pmeilleur_[self.i_, self.COL_INDICE_] = ind
		self.Pmeilleur_[self.i_, self.COL_CLASSE_] = self.ClassesPrototypes_[ind]
		self.Pmeilleur_[self.i_, self.COL_DISTANCE_] = D[ind]

		return self.Pmeilleur_[self.i_, :]

	def chercherPsecond(self, x, D, classePmeilleur):
		try:
			# les protos d'une classe différente de celle de Pmeilleur
			valid_id = np.where(self.ClassesPrototypes_[:self.p_,] != classePmeilleur)[0]
			
			# si on n'a pas troupé : il n'y en a pas !
			if not valid_id.size:
				raise ValueError('Pas de prototyes pour Psecond')
			
			# s'il y en a, on prend le plus proche de X
			ind = valid_id[D[valid_id].argmin()] 
			self.Psecond_[self.i_, self.COL_INDICE_] = ind
			self.Psecond_[self.i_, self.COL_CLASSE_] = self.ClassesPrototypes_[ind]
			self.Psecond_[self.i_, self.COL_DISTANCE_] = D[ind]

		except ValueError :
			self.Psecond_[self.i_, self.COL_INDICE_] = np.nan
			self.Psecond_[self.i_, self.COL_CLASSE_] = np.nan
			self.Psecond_[self.i_, self.COL_DISTANCE_] = np.nan

		return self.Psecond_[self.i_, :]

	def retirerNaN(self):
		self.Prototypes_ = self.Prototypes_[:self.p_, :]
		self.ClassesPrototypes_ = self.ClassesPrototypes_[:self.p_]
		pass

	''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%ù
			IMPORT - EXPORT
	'''

	def printInfosRun(self):
		"""
		Affiche le numéro de passe et le nombre de prototypes, uniquement
		si le mode verbose est à True
		"""
		if self.verbose:
			print ('Passe ', self.i_, ' // ', self.p_, ' prototypes')

	def exportModele(self, save_path):
		hf = h5py.File(save_path, 'w')

		gp = hf.create_group("Modele")
		gp.create_dataset('Classes', data=self.ClassesPrototypes_)
		gp.create_dataset('Prototypes', data=self.Prototypes_)
		gp.create_dataset('Conditions', data = self.Conditions_, dtype='bool')

		gp = hf.create_group("Hyperparametres")
		gp.create_dataset('s_inf', data=self.s_inf)
		gp.create_dataset('s_conf', data=self.s_conf)
		gp.create_dataset('alpha', data=self.alpha)

		gp = hf.create_group("Divers")
		gp.create_dataset('PMeilleur', data = self.Pmeilleur_)
		gp.create_dataset('PSecond', data = self.Psecond_)

		hf.close()

	def importModele(self, save_path):
		hf = h5py.File(save_path, 'r')
		self.ClassesPrototypes_ = np.array(hf.get('/Modele/Classes'))
		self.Prototypes_ = np.array(hf.get('/Modele/Prototypes'))
		self.s_inf = np.array(hf.get('/Hyperparametres/s_inf'))
		self.s_conf = np.array(hf.get('/Hyperparametres/s_conf'))
		self.alpha = np.array(hf.get('/Hyperparametres/alpha'))

		self.initCols()
		self.calculerArbres()
		hf.close()

	def exportClassif(self, chemin):
		hf = h5py.File(chemin, 'w')
		gp = hf.create_group("Reponses")
		gp.create_dataset('Cond. de classif', data = self.Conditions_Classif_)

		gp.create_dataset('Brut/Non reponses', data = self.nonRep_)
		gp.create_dataset('Brut/Bonnes reponses', data = self.bonRep_)
		gp.create_dataset('Brut/Mauvaises reponses', data = self.mauvRep_)
		gp.create_dataset('Brut/Bon intervalle', data = self.bonIntervalle_)
		
		gp.create_dataset('Classif/Type Reponses', data = self.Reponse_)
		gp.create_dataset('Classif/Verite', data = self.True_)
		gp.create_dataset('Classif/Prediction', data = self.Pred_)
		

		gp.create_dataset('Taux/Bonnes reponses', data = self.bonRep_*100.0/(len(self.True_)))
		gp.create_dataset('Taux/Mauvaises reponses', data = self.mauvRep_*100.0/(len(self.True_)))
		gp.create_dataset('Taux/Non reponses', data = self.nonRep_*100.0/(len(self.True_)))
		gp.create_dataset('Taux/Bon intervalle', data = self.bonIntervalle_*100.0/(len(self.True_)))

		
		gp = hf.create_group("Erreur")
		gp.create_dataset('RMSE', data = self.rmse_)
		gp.create_dataset('MAE', data = self.mae_)

		'''
		gp = hf.create_group("Evaluation BDI")
		gp.create_dataset('Par score/Specificite', data = np.transpose(np.vstack((self.ClassesBDI_, self.specificiteBDI_))))
		gp.create_dataset('Par score/Sensibilite', data = np.transpose(np.vstack((self.ClassesBDI_, self.sensibiliteBDI_))))
		gp.create_dataset('Par score/Accuracy', data = np.transpose(np.vstack((self.ClassesBDI_, self.accuracyBDI_))))
		gp.create_dataset('Par score/FMesure', data = np.transpose(np.vstack((self.ClassesBDI_, self.fmesureBDI_))))
		gp.create_dataset('Moyennes/M_specificite', data = np.mean(self.specificiteBDI_))
		gp.create_dataset('Moyennes/M_sensibilite', data = np.mean(self.sensibiliteBDI_))
		gp.create_dataset('Moyennes/M_fmesure', data = np.mean(self.fmesureBDI_))
		gp.create_dataset('Moyennes/M_accuracy', data = np.mean(self.accuracyBDI_))
		'''
		gp = hf.create_group("Evaluation Scores")
		gp.create_dataset('Par score/Specificite', data = np.transpose(np.vstack((self.Classes_, self.specificite_))))
		gp.create_dataset('Par score/Sensibilite', data = np.transpose(np.vstack((self.Classes_, self.sensibilite_))))
		gp.create_dataset('Par score/Accuracy', data = np.transpose(np.vstack((self.Classes_, self.accuracy_))))
		gp.create_dataset('Par score/FMesure', data = np.transpose(np.vstack((self.Classes_, self.fmesure_))))
		gp.create_dataset('Moyennes/M_specificite', data = np.mean(self.specificite_))
		gp.create_dataset('Moyennes/M_sensibilite', data = np.mean(self.sensibilite_))
		gp.create_dataset('Moyennes/M_fmesure', data = np.mean(self.fmesure_))
		gp.create_dataset('Moyennes/M_accuracy', data = np.mean(self.accuracy_))


		gp = hf.create_group("Confusion Scores")
		gp.create_dataset('Matrice', data = self.MatriceConfScores_)
		gp.create_dataset('Vrais positifs', data = self.VP_)
		gp.create_dataset('Vrais negatifs', data = self.VN_)
		gp.create_dataset('Faux positifs', data = self.FP_)
		gp.create_dataset('Faux negatifs', data = self.FN_)

		gp = hf.create_group("Confusion BDI")
		gp.create_dataset('Matrice', data = self.MatriceConfBDI_)
		gp.create_dataset('Vrais positifs', data = self.VP_BDI_)
		gp.create_dataset('Vrais negatifs', data = self.VN_BDI_)
		gp.create_dataset('Faux positifs', data = self.FP_BDI_)
		gp.create_dataset('Faux negatifs', data = self.FN_BDI_)

		gp = hf.create_group("Divers")
		gp.create_dataset('Distances Pmeilleur', data = self.PM_)
		gp['Distances Pmeilleur'].attrs['Colonne_0'] = 'indice du meilleur prototype'
		gp['Distances Pmeilleur'].attrs['Colonne_1'] = 'classe du meilleur prototype'
		gp['Distances Pmeilleur'].attrs['Colonne_2'] = 'distance au meilleur prototype'
		gp.create_dataset('Distances Psecond', data = self.PS_)

		hf.close()

	''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			CLASSIFICATION
	'''
	def score(self, X, y):
		self.True_ = y
		self.Conditions_Classif_ = np.zeros((len(X), 3), dtype=np.int)

		self.calculerArbres()
		self.calculerPmeilleur(X)
		self.calculerPsecond(X)
		self.classifier(X, y)
		self.stat()
		self.evaluerClassification()
 
		self.rmse_= self.calcErreur(self.True_, self.Pred_)
		self.mae_ = self.calcErreur(self.True_, self.Pred_, err='mae')

		return np.mean(self.accuracy_)

		#return self.rmse_



	def predict(self, X):
		self.Conditions_Classif_ = np.zeros((len(X), 3), dtype=np.int)

		self.calculerArbres()
		self.calculerPmeilleur(X)
		self.calculerPsecond(X)
		self.classifier(X)

		if len(self.Pred_ == 1):
			return self.Pred_[0]
		else:
			return self.Pred_

	def calculerArbres(self):
		'''
		prototypes[i] : le ie prototype
		classes[i] : la classe de prototypes[i]

		Calcule autant d'arbres qu'il y a de classes + 1
		arbre avec toutes les classes
		'''
		self.Arbres_ = []
		self.indexArbres_ = np.unique(self.ClassesPrototypes_)
		self.P_ = KDTree(self.Prototypes_, metric=self.distanceType)

		for u in self.indexArbres_:
			indProtos = np.where(self.ClassesPrototypes_ != u)[0]
			protos = self.Prototypes_[indProtos, :]

			tree = KDTree(protos, metric=self.distanceType)
			self.Arbres_.append(tree)

		return self.P_, self.indexArbres_

	def calculerPmeilleur(self, X):
		'''
		distance[i] : la distance entre l'ex test[i] et son Pmeilleur
		indice[i] : l'indice du Pmeilleur de test[i] tel que protos[indice[i]]
		est le Pmeilleur de test[i] 
		'''

		try:
			Distance, Indice = self.P_.query(X)
			 
		except ValueError:

			Distance, Indice = self.P_.query(X.reshape(1, -1))

		ClassePmeilleur = self.ClassesPrototypes_[Indice[:, 0]] 
		self.PM_ = np.hstack((Indice, ClassePmeilleur, Distance))

		return self.PM_

	def calculerPsecond(self, X):
		'''
		classesTest[i] : la classe de l'ex de test test[i]
		'''
		self.PS_ = np.zeros((len(self.PM_), 3))
		
		try:
			for i in range(len(X)):
				ind = np.where(self.indexArbres_ == self.PM_[i, self.COL_CLASSE_])[0][0]
				
				a = self.Arbres_[ ind ]
				distance, indice = a.query(X[i, :].reshape(1, -1))
				self.PS_[i] = [indice, 0, distance]

		except IndexError:
			ind = np.where(self.indexArbres_ == self.PM_[0, self.COL_CLASSE_])[0][0]
			a = self.Arbres_[ ind ]
			distance, indice = a.query(X.reshape(1, -1))

		return self.PS_

	def classifier(self, X, y=[]):

		self.Reponse_ = np.zeros((len(self.PM_), 1))
		#Important d'avoir une liste, et pas un tableau à n lignes ou colones
		#faisait planter le calcul de l'erreur
		self.Pred_ = np.zeros((len(self.PM_),))*-1
		#self.True_ = y 

		 
		if y == []:
			y = np.zeros((len(X),))

		for i in range(len(self.PM_)):
			d = abs(self.PM_[i, self.COL_DISTANCE_] - self.PS_[i, self.COL_DISTANCE_])
			
			self.Conditions_Classif_[i] = [self.PM_[i, self.COL_DISTANCE_] > self.s_inf, d < self.s_conf, self.PM_[i, self.COL_CLASSE_] == y[i]]

			if self.PM_[i, self.COL_DISTANCE_] > self.s_inf or d < self.s_conf: 

				self.Reponse_[i] = -1
			elif np.abs(self.PM_[i, self.COL_CLASSE_] - y[i]) == 0:
				self.Reponse_[i] = 1
				#self.Pred_[i] = self.PM_[i, self.COL_CLASSE_]
			else :
				self.Reponse_[i] = 0

			self.Pred_[i] = self.PM_[i, self.COL_CLASSE_]
 
		return self.Reponse_[i]

	def determinerIntervalleBDI(self, vecteur):
		''' Renvoie un vecteur où chaque valeur est l'intervalle BDI (0, 1, 2, 3)
		du score BDI correspondant dans le param. vecteur

		'''
		intervalles = np.transpose(np.array([[0, 14, 20, 29], [13, 19, 28, 63]]))
		intBDI = np.zeros((len(vecteur), ), dtype = int)

		for i in range(len(vecteur)):
			for c in range(4):
				if intervalles[c, 0] <= vecteur[i] <= intervalles[c, 1] :
					intBDI[i] = c

		# retour 1, 2, 3, 4
		return intBDI+1

	def nbMemeIntervalle(self, verite, prediction):
		''' Renvoie le nombre d'éléments de verite et prediction qui sont dans le
		m^eme intervalle BDI
		'''
		intBDI_verite = self.determinerIntervalleBDI(verite)
		intBDI_pred = self.determinerIntervalleBDI(prediction)

		idx = np.where(self.Reponse_ != -1)[0]

		#intBDI_verite = intBDI_verite[idx]
		#intBDI_pred = intBDI_pred[idx]

		diff = intBDI_verite - intBDI_pred;
		unique, count = np.unique(diff, return_counts = True)
		count = dict(zip(unique, count))

		if 0 in count: 
			return count.get(0) 
		else:
			return -1
		

	''' %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			EVALUATION
	'''
	def stat(self):
		unique, counts = np.unique(self.Reponse_, return_counts=True)
		# print dict(zip(unique, counts))

		self.mauvRep_ = 0
		self.bonRep_ = 0
		self.nonRep_ = 0
		
		for i in range(len(unique)):
			if unique[i] == 0:
				self.mauvRep_= counts[i]
			elif unique[i] == 1:
				self.bonRep_ = counts[i]
			elif unique[i] == -1:
				self.nonRep_ = counts[i]

		#self.bonIntervalle_ = self.nbMemeIntervalle(self.True_, self.Pred_)

	def plotFace(self, vecteur):
		X = vecteur[:, :68] 
		Y = vecteur[:, 68:] 
		plt.scatter(X, Y) 
		plt.show()

	def calcErreur(self, T, P, err='rmse'):

		if err == 'rmse':

			return  math.sqrt( np.sum((T-P)**2) / len(T) )
		elif err == 'mae':
			#print np.sum(Reponses[:, 0] - Reponses[:, 1])
			return np.sum(  abs(T - P)  ) /len(T)

	def evaluerClassification(self):
		self.MatriceConfBDI_ = self.calculerConfusion(opt='BDI')
		self.MatriceConfScores_ = self.calculerConfusion(opt='score')

		COL_CLASSE_ = 0
		COL_VALEUR = 1


		self.Classes_ = range(63)
		self.VP_, self.VN_, self.FP_, self.FN_ = self.calculerTest(self.MatriceConfScores_)

		self.specificite_ = self.statMesures(self.VP_, self.VN_, self.FP_, self.FN_, self.Classes_,'specificite')
		self.sensibilite_ = self.statMesures(self.VP_, self.VN_, self.FP_, self.FN_, self.Classes_,'sensibilite')
		
		self.accuracy_ = self.statMesures(self.VP_, self.VN_, self.FP_, self.FN_, self.Classes_,'accuracy')
		self.fmesure_ = self.statMesures(self.VP_, self.VN_, self.FP_, self.FN_, self.Classes_,'fmesure')

		
		self.VP_BDI_, self.VN_BDI_, self.FP_BDI_, self.FN_BDI_ = self.calculerTest(self.MatriceConfBDI_)
		self.ClassesBDI_ = range(1, 5)
		self.specificiteBDI_ = self.statMesures(self.VP_BDI_, self.VN_BDI_, self.FP_BDI_, self.FN_BDI_, self.ClassesBDI_, 'specificite')
		self.sensibiliteBDI_ = self.statMesures(self.VP_BDI_, self.VN_BDI_, self.FP_BDI_, self.FN_BDI_, self.ClassesBDI_,'sensibilite')
		self.accuracyBDI_ = self.statMesures(self.VP_BDI_, self.VN_BDI_, self.FP_BDI_, self.FN_BDI_, self.ClassesBDI_,'accuracy')
		self.fmesureBDI_ = self.statMesures(self.VP_BDI_, self.VN_BDI_, self.FP_BDI_, self.FN_BDI_, self.ClassesBDI_,'fmesure')
		
	def statMesures(self, VP, VN, FP, FN, classes, mesure='accuracy'):
		''' Une mesure est produite par classe. Il y a donc autant de valeurs
		dans Mesure que de classes. La mesure est définie par le param. accuracy
		'''
		Mesure = np.zeros((len(VP), ), dtype = float) 


		#n = len(classes) 
		n = np.amax([len(VP), len(FP), len(VN), len(FN)])

		for i in range(n):
			# spécificité

			try:
				if mesure == 'specificite':
					if (VN[i] + FP[i]) == 0: 
						raise ValueError('Division par zero')
					Mesure[i] = VN[i] / (VN[i] + FP[i])

				elif mesure == 'sensibilite':
					if (VP[i] + FN[i]) == 0:
						raise ValueError('Division par zero')
					Mesure[i] = VP[i] / (VP[i] + FN[i])
					#print Mesure[i], '=', VP[i], FN[i]

				elif mesure == 'accuracy':
					if (VP[i] + VN[i] + FP[i] + FN[i]) == 0:
						raise ValueError('Division par zero')
					Mesure[i] = (VP[i] + VN[i]) / (VP[i] + VN[i] + FP[i] + FN[i])
				elif mesure == 'fmesure':
					if (2*VP[i] + FP[i] + FN[i]) == 0:
						raise ValueError('Division par zero')
					Mesure[i] = (2*VP[i]) / (2*VP[i] + FP[i] + FN[i])

			except ValueError:
				Mesure[i] = 0

		return Mesure

	def calculerConfusion(self, opt='BDI'):

		if opt == 'BDI':
			V = self.determinerIntervalleBDI(self.True_)
			P = self.determinerIntervalleBDI(self.Pred_)
			return confusion_matrix(V, P)
		elif opt == 'score':
			return confusion_matrix(self.True_, self.Pred_, labels=range(3))

	def calculerTest(self, matriceConfusion):
		VP = np.zeros((len(matriceConfusion),))
		VN = np.zeros((len(matriceConfusion),))
		FP = np.zeros((len(matriceConfusion),))
		FN = np.zeros((len(matriceConfusion),))

		for i in range(len(matriceConfusion)):
			# DIAGONALE
			VP[i] = matriceConfusion[i, i]

			# LA LIGNE, SAUF LA CELLULE DE LA DIAG, I.E SAUF VP
			FN[i] = np.abs(np.sum(matriceConfusion[i]) - VP[i])

			# LA COLONNE, SAUF LA CELLULE DE LA DIAG, I.E SAUF VP
			FP[i] = np.abs(np.sum(matriceConfusion[:, i]) - VP[i])

			# LE RESTE
			VN[i] = np.abs(np.sum(np.sum(matriceConfusion)) - (VP[i] + FN[i] + FP[i]))


		return VP, VN, FP, FN
		
	def monHamming(self, A, b): 

		if not (A.shape[1] == b.shape[0] and len(b.shape) == 1):
			print("pb de taille dans hamming : {} vs {}".format(A.shape[1], b.shape[1]))
			exit()


		h = np.zeros((A.shape[0], ))
		for i in range(0, A.shape[0]):
			h[i] = np.sum(A[i, :] != b)

		return h
'''
base = '/home/ss/Documents/programmation/DATA/old_repere/'
data = base + 'total/total_V3.h5'

test = base + 'devel_northwind/develNorthwind_V4.h5'
 
taille = 203809
D = ci2_data.Data()
donnees = D.chargerHDF5(data)
X = donnees[:taille, 3:]
y = donnees[:taille, 2]

#donnees = D.chargerHDF5(test)
X_test = donnees[taille:, 3:]
y_test = donnees[taille:, 2]

A = ClassifieurIncremental(s_inf=70, s_conf=0.1, alpha=0.1, verbose=True)
print A.fit(X, y)
print A.score(X_test, y_test)
print len(A.Prototypes_), 'prototypes'
print A.rmse_, A.mae_
print A.mauvRep_, A.bonRep_, A.nonRep_,

A.exportModele('./protos.h5')

np.savetxt('out_pred.csv', A.Pred_, fmt='%.2f')
np.savetxt('out_true.csv', A.True_, fmt='%.2f')
'''

'''
A = ClassifieurIncremental()
nbEx = 50

X = np.random.rand(nbEx, 150)
Y = np.random.randint(1, high=8, size=nbEx)

print A.fit(X, Y) 
print A.predict(X, Y)'''