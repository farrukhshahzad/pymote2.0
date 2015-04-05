from pymote import *
from networkx.generators.classic import star_graph,balanced_tree,barbell_graph
from networkx.generators.social import florentine_families_graph
from networkx import Graph, is_connected,draw
from networkx.algorithms.operators import union_all, disjoint_union_all


try:
            from matplotlib import pyplot as plt
except ImportError:
            raise ImportError("Matplotlib required for show()")



from pymote import energy
en = energy.EnergyModel()
en.TR_RATE = 200
print en.TR_RATE, en, en.energy, en.increase_energy(),\
    en.decrease_tx_energy(100), en.decrease_rx_energy(100), en

from pymote import mobility
en = mobility.MobilityModel(mobile_type=1)
#en.HEADING = -en.HEADING
print en.VELOCITY, en.HEADING, en, en.drift(), en.have_moved()

from pymote import propagation
en = propagation.PropagationModel(propagation_type=1)
en2 = propagation.PropagationModel(propagation_type=1)
pr = en.get_power_ratio(d=405, p_tx=1)
print en, pr, propagation.PropagationModel.pw_to_dbm(pr), en.is_rx_ok(), \
    en.free_space_distance(p_rx=2.354466826905901e-09)

print en2, en2.cross_over_distance(), en2.get_power_ratio(d=205), \
      propagation.PropagationModel.dbm_to_pw(11.6), en2.two_ray_ground_distance()

en2 = propagation.PropagationModel(propagation_type=2)
pr = en2.shadowing(d=405)
print en2, pr, propagation.PropagationModel.pw_to_dbm(pr), en2.shadowing_rssi(d=405)

propagation.PropagationModel.P_TX = 0.0144
en2 = propagation.PropagationModel(propagation_type=2)
pr = en2.shadowing(d=405)
print en2, pr, propagation.PropagationModel.pw_to_dbm(pr), en2.shadowing_rssi(d=405)
