#### Baselines: for running rl card agents against itself
```python evaluate.py --landlord rlcard --landlord_up rlcard --landlord_down rlcard```

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
```python evaluate.py --landlord rlcardV2 --landlord_up rlcard --landlord_down rlcard```

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
##### Note: this is playing against the implementation that has broken picker for pair chains (requiring 5 pairs in a row).

```python evaluate.py --landlord rlcardV2 --landlord_up rlcard --landlord_down rlcard```

Attempt 1
WP results:
landlord : Farmers - 0.5564 : 0.4436
ADP results:
landlord : Farmers - 0.2678 : -0.2678

Attempt 2
WP results:
landlord : Farmers - 0.5556 : 0.4444
ADP results:
landlord : Farmers - 0.2936 : -0.2936

Attempt 3
WP results:
landlord : Farmers - 0.5551 : 0.4449
ADP results:
landlord : Farmers - 0.2836 : -0.2836

##### Playing against working picker for pair chains.

```python evaluate.py --landlord rlcardV2 --landlord_up rlcard --landlord_down rlcard```

Attempt 1
WP results:
landlord : Farmers - 0.5505 : 0.4495
ADP results:
landlord : Farmers - 0.2276 : -0.2276

Attempt 2
WP results:
landlord : Farmers - 0.5482 : 0.4518
ADP results:
landlord : Farmers - 0.2132 : -0.2132

Attempt 3
WP results:
landlord : Farmers - 0.5515 : 0.4485
ADP results:
landlord : Farmers - 0.2524 : -0.2524

##### V2 with picking lowest kickers from solo and pairs for trios

```python evaluate.py --landlord rlcardV2 --landlord_up rlcard --landlord_down rlcard```

Attempt 1
WP results:
landlord : Farmers - 0.6388 : 0.3612
ADP results:
landlord : Farmers - 0.7798 : -0.7798

Attempt 2
WP results:
landlord : Farmers - 0.638 : 0.362
ADP results:
landlord : Farmers - 0.7788 : -0.7788

Attempt 3
WP results:
landlord : Farmers - 0.6328 : 0.3672
ADP results:
landlord : Farmers - 0.7398 : -0.7398


## Next steps
Try adding kickers to triples.
Try prioritizing low sets of trios