import random
from typing import Any

class Agent:
    """Fixed Q-table agent for discrete environments (e.g., FrozenLake 8x8).

    - Selects the action with the highest Q-value for a given discrete state.
    - Falls back to a random action when the state is invalid or ties exist beyond
      a small tolerance.
    """
    TOLERANCE_OPTIMALE = 1e-6

    def __init__(self, env: Any) -> None:
        self.environnement = env
        
        # Table de valeurs Q (FrozenLake 8x8, 64 états) 
        self.table_q = [[0.579822, 0.581208, 0.581296, 0.581987], [0.587686, 0.593834, 0.596789, 0.592168], [0.501263, 0.593919, 0.511577, 0.504278], [0.512099, 0.523788, 0.528393, 0.51851], [0.539321, 0.537193, 0.546807, 0.535641], [0.561947, 0.537497, 0.577568, 0.570087], [0.589724, 0.591687, 0.507891, 0.500561], [0.508826, 0.595327, 0.521975, 0.507772], [0.572792, 0.578569, 0.572783, 0.58057], [0.578419, 0.57911, 0.588616, 0.590861], [0.578901, 0.584723, 0.589491, 0.50777], [0.306092, 0.271261, 0.192677, 0.425032], [0.424573, 0.424439, 0.430245, 0.447883], [0.462084, 0.46709, 0.475146, 0.484985], [0.503742, 0.505556, 0.511978, 0.495414], [0.519936, 0.524105, 0.535583, 0.512079], [0.563731, 0.557041, 0.566048, 0.568274], [0.567797, 0.557717, 0.562273, 0.572242], [0.560071, 0.203003, 0.212386, 0.288844], [0.0, 0.0, 0.0, 0.0], [0.165207, 0.26189, 0.397987, 0.31324], [0.244773, 0.310855, 0.342719, 0.409621], [0.485391, 0.508537, 0.509183, 0.505618], [0.546134, 0.536503, 0.554996, 0.533032], [0.346979, 0.331184, 0.345461, 0.352486], [0.317503, 0.310109, 0.327917, 0.339505], [0.294616, 0.284331, 0.223261, 0.301758], [0.124132, 0.19764, 0.044231, 0.211458], [0.27847, 0.184013, 0.206848, 0.155309], [0.0, 0.0, 0.0, 0.0], [0.253917, 0.314493, 0.509288, 0.391086], [0.548566, 0.567276, 0.588164, 0.551287], [0.302249, 0.290915, 0.311257, 0.320873], [0.256231, 0.164413, 0.187382, 0.289321], [0.152053, 0.043329, 0.137134, 0.184962], [0.0, 0.0, 0.0, 0.0], [0.153494, 0.167452, 0.231804, 0.199302], [0.133823, 0.305486, 0.264708, 0.203588], [0.158017, 0.35627, 0.308308, 0.440469], [0.557755, 0.622413, 0.643864, 0.588574], [0.279294, 0.153106, 0.229827, 0.204378], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.003946, 0.05527, 0.065257, 0.067797], [0.122664, 0.093876, 0.154395, 0.159344], [0.186022, 0.088138, 0.128625, 0.162662], [0.0, 0.0, 0.0, 0.0], [0.551998, 0.653939, 0.713163, 0.457257], [0.261783, 0.140028, 0.180178, 0.222388], [0.0, 0.0, 0.0, 0.0], [0.018482, 0.00597, 0.013521, 0.000941], [0.019144, 0.001296, 0.007722, 0.02424], [0.0, 0.0, 0.0, 0.0], [0.069913, 0.032439, 0.100779, 0.08485], [0.0, 0.0, 0.0, 0.0], [0.609852, 0.818184, 0.910502, 0.412618], [0.257912, 0.225964, 0.235324, 0.201547], [0.144473, 0.182086, 0.098432, 0.09591], [0.081182, 0.050158, 0.026268, 0.047981], [0.0, 0.0, 0.0, 0.0], [0.000809, 0.009078, 0.001028, 0.001663], [0.024167, 0.064922, 0.085727, 0.055384], [0.026923, 0.234484, 0.196607, 0.19], [0.0, 0.0, 0.0, 0.0]]

        self.limite_etats = len(self.table_q)
        self.limite_actions = len(self.table_q[0]) if self.limite_etats > 0 else env.action_space.n

    def choose_action(self, observation: Any, **kwargs) -> int:
        """Choose an action for a given observation.

        Returns an integer action index. If the observation cannot be converted
        to a valid state index, a random action is sampled from the environment.
        """
        try:
            identifiant_etat = int(observation)
        except Exception:
            return self.environnement.action_space.sample()

        if identifiant_etat < 0 or identifiant_etat >= self.limite_etats:
            return self.environnement.action_space.sample()

        ligne_q = self.table_q[identifiant_etat]
        
        if not ligne_q:
             return self.environnement.action_space.sample()

        valeur_max = max(ligne_q)
        
        candidats_optimaux = []
        
        for index_action, valeur_q in enumerate(ligne_q):
            if abs(valeur_q - valeur_max) < self.TOLERANCE_OPTIMALE:
                candidats_optimaux.append(index_action)

        if candidats_optimaux:
            return random.choice(candidats_optimaux)
        else:
            return self.environnement.action_space.sample()
