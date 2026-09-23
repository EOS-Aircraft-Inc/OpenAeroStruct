"""WingCalc's wing weight build-up, ported so OpenAeroStruct can compute it alone.

``deck_inputs``  the deck WingCalc reads, with the two overrides ``write_deck`` applies
``buildup``      every term of ``weight_calc.compute_wing_weight``, on OAS geometry
``component``    the OpenMDAO wrapper that puts it inside the optimization
"""
