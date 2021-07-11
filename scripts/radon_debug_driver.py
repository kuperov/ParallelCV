#!/usr/bin/env python

from ploo import LGO
from ploo.models import RadonCountyIntercept

model = RadonCountyIntercept()
post = model.inference(draws=2e3, chains=4, warmup_steps=1e3, seed=42)

lgo_scheme = LGO(shape=model.log_radon.shape, group_ids=model.county_index)
cv = post.cross_validate(cv_scheme=lgo_scheme)
print(cv)

# so we can download elpds by group and compare with psis
import pandas as pd
df = pd.DataFrame({'elpds': cv.fold_elpds})
df.to_csv('radon_elpds.csv')
