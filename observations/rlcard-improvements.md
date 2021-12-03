#### Baselines: for running rl card agents against itself
Command: python evaluate.py --landlord rlcard --landlord_up rlcard --landlord_down rlcard

Attempt 1
WP results:
landlord : Farmers - 0.4621 : 0.5379
ADP results:
landlord : Farmers - -0.1604 : 0.1604

Attempt 2
landlord : Farmers - 0.4569 : 0.5431
ADP results:
landlord : Farmers - -0.2156 : 0.2156
Attempt 3
WP results:
landlord : Farmers - 0.4616 : 0.5384
ADP results:
landlord : Farmers - -0.1776 : 0.1776

---

#### Agent that prioritizes chains (Note: pair chains is broken for these observations)
Command: python evaluate.py --landlord rlcardV2 --landlord_up rlcard --landlord_down rlcard

Attempt 1
WP results:
landlord : Farmers - 0.4739 : 0.5261
ADP results:
landlord : Farmers - -0.1156 : 0.1156

Attempt 2
WP results:
landlord : Farmers - 0.4748 : 0.5252
ADP results:
landlord : Farmers - -0.1182 : 0.1182


Attempt 3
WP results:
landlord : Farmers - 0.4742 : 0.5258
ADP results:
landlord : Farmers - -0.108 : 0.108

---

#### Agent that prioritizes chains from combos and picks pair_chains that are non-disruptive of large straights.
Note: this is playing against the implementation that has broken picker for pair chains (requiring 5 pairs in a row).
Attempt 1
WP results:
landlord : Farmers - 0.4961 : 0.5039
ADP results:
landlord : Farmers - 0.0268 : -0.0268

Attempt 2
WP results:
landlord : Farmers - 0.4907 : 0.5093
ADP results:
landlord : Farmers - -0.022 : 0.022

Attempt 3
WP results:
landlord : Farmers - 0.4951 : 0.5049
ADP results:
landlord : Farmers - 0.0092 : -0.0092


#### Agent that prioritizes chains from combos and picks pair_chains that are non-disruptive of large straights.
Playing against working picker for pair chains.

Attempt 1
WP results:
landlord : Farmers - 0.4889 : 0.5111
ADP results:
landlord : Farmers - -0.0312 : 0.0312

Attempt 2
WP results:
landlord : Farmers - 0.4893 : 0.5107
ADP results:
landlord : Farmers - -0.0284 : 0.0284

Attempt 3
WP results:
landlord : Farmers - 0.4881 : 0.5119
ADP results:
landlord : Farmers - -0.0442 : 0.0442
