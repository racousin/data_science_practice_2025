import random

class Agent:
    Q_TOLERANCE = 1e-6

    def __init__(self, env):
        self.jeu_environnement = env
        
        # NOUVELLE Q-TABLE OPTIMISÉE pour FrozenLake 4x4 (16 états). Taux de succès élevé.
        self.politique_reference = [[0.06509419, 0.08053676, 0.09100769, 0.08678241], [0.03859664, 0.06456071, 0.08889416, 0.07685244], [0.05436662, 0.1176503 , 0.15570228, 0.13429393], [0.09033282, 0.0       , 0.0       , 0.0       ], [0.08118084, 0.14798604, 0.10113192, 0.04028308], [0.0       , 0.0       , 0.0       , 0.0       ], [0.11663737, 0.22271633, 0.21856272, 0.17066804], [0.0       , 0.0       , 0.0       , 0.0       ], [0.14488426, 0.35406453, 0.22013867, 0.09886475], [0.19894178, 0.43981446, 0.46162136, 0.3503144 ], [0.30154832, 0.5367018 , 0.58988691, 0.44497645], [0.0       , 0.0       , 0.0       , 0.0       ], [0.0       , 0.0       , 0.0       , 0.0       ], [0.3802956 , 0.69708948, 0.76077884, 0.57500355], [0.65586616, 0.81434316, 0.87979685, 0.81189434], [0.0       , 0.0       , 0.0       , 0.0       ]]
        
        self.nombre_etats = len(self.politique_reference) 
        self.nombre_actions = len(self.politique_reference[0]) if self.nombre_etats > 0 else env.action_space.n

    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None):
        etat_courant_idx = int(observation)

        if etat_courant_idx < 0 or etat_courant_idx >= self.nombre_etats:
            return random.randint(0, self.nombre_actions - 1)

        valeurs_q_etat = self.politique_reference[etat_courant_idx]
        valeur_q_maximale = max(valeurs_q_etat)
    
        actions_optimales = []
        for index_action, valeur_q in enumerate(valeurs_q_etat):
            if abs(valeur_q - valeur_q_maximale) < self.Q_TOLERANCE:
                actions_optimales.append(index_action)

        if actions_optimales:
            return actions_optimales[0]
        else:
            return random.randint(0, self.nombre_actions - 1)