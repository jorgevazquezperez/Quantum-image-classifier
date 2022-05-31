"""
=================================================================
Quantum image classifier module (:mod:`quantum_image_classifier`)
=================================================================
.. currentmodule:: quantum_image_classifier

This is the quantum image classifier implemented by Jorge Vázquez Pérez. The set of algorithms here
implemented can be used with :mod:`~quantum_image_classifier.NAME_OF_THE_ALGORITHM`.
"""

from .classifier_algorithms import NearestCentroid
from .data_treatment import data_generator, data_loader, QuantumAE, QuantumAE2
from .encoding import UnaryLoader, PhaseEncoding
from .gates import RBS, iRBS
from .visuals import error_graph, variance_error_graph