#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from scipy import signal
from run_regime_discovery_audit import PERIODS,candidate_specs,fit_candidate,load_data

CONT=["rv_20d","rv_ratio_5_20","rv_ratio_20_60","vix_close","vix_rv_spread","spy_drawdown_20d"]
KEEP={"risk4_k2","risk4_k3","state8_k2","vol_level_x_acceleration_grid"}
BANDS=[(20,60),(60,125),(125,250),(250,500),(500,1000),(1000,1500)]

def args():
 p=argparse.ArgumentParser();p.add_argument("--data",type=Path,default=Path("data/processed/phase2_spy_vix_volatility.csv"));p.add_argument("--outdir",type=Path,default=Path("results/diagnostics/frequency_structure"));p.add_argument("--surrogates",type=int,default=500);p.add_argument("--block-size",type=int,default=60);p.add_argument("--max-acf-lag",type=int,default=1000);p.add_argument("--seed",type=int,default=20260704);return p.parse_args()

def mask(df,p):
 lo,hi=PERIODS[p];return (df.date>=lo)&(df.date<=hi)

def add_series(df,seed):
 df=df.copy();df["vix_rv_spread"]=np.log(np.maximum(df.vix_close,1e-12)/100)-np.log(np.maximum(df.rv_20d,1e-12))
 for i,s in enumerate(candidate_specs()):
  if s.name not in KEEP: continue
  c=f"state__{s.name}";df[c]=fit_candidate(df,s,seed+i)
  for st in sorted(int(x) for x in df[c].dropna().unique()): df[f"{c}__is_{st}"]=(df[c]==st).astype(float)
 return df

def acf_stats(x,L):
 x=x-np.mean(x);n=len(x)
 if n<3 or np.var(x)==0:return dict(acf_1=np.nan,lag_below_1_over_e=np.nan,first_zero_crossing=np.nan,integrated_positive_acf_time=np.nan)
 m=1<<int(np.ceil(np.log2(2*n-1)));f=np.fft.rfft(x,m);a=np.fft.irfft(f*np.conj(f),m)[:n]/np.arange(n,0,-1);a=(a/a[0])[:min(L,n-1)+1]
 b=np.where(a[1:]<np.exp(-1))[0];z=np.where(a[1:]<=0)[0];zc=int(z[0]+1) if len(z) else np.nan;pos=a[1:int(zc)] if np.isfinite(zc) else a[1:];pos=pos[(pos>0)&np.isfinite(pos)]
 return dict(acf_1=float(a[1]),lag_below_1_over_e=int(b[0]+1) if len(b) else np.nan,first_zero_crossing=zc,integrated_positive_acf_time=float(1+2*np.sum(pos)))

def psd(x):
 if len(x)<64 or np.var(x)==0:return np.array([]),np.array([])
 n=min(1024,max(64,2**int(np.floor(np.log2(len(x)/2)))));n=min(n,len(x));f,p=signal.welch(x,fs=1,window="hann",nperseg=n,noverlap=n//2,detrend="linear");q=f>0;return f[q],p[q]

def bpower(f,p,lo,hi):
 if not len(f):return np.nan
 per=1/f;m=(per>=lo)&(per<hi)
 if not m.any():return np.nan
 return float(np.trapz(p[m],f[m])) if m.sum()>1 else float(p[m][0])

def peak(f,p):
 if not len(f):return np.nan,np.nan
 per=1/f;m=(per>=20)&(per<=1500)
 if not m.any():return np.nan,np.nan
 j=np.flatnonzero(m)[np.argmax(p[m])];return float(per[j]),float(p[j])

def perm(x,B,rng):
 bs=[x[i:min(i+B,len(x))] for i in range(0,len(x),B)];o=rng.permutation(len(bs));return np.concatenate([bs[i] for i in o])

def main():
 a=args();a.outdir.mkdir(parents=True,exist_ok=True);df=add_series(load_data(a.data),a.seed);series_names=CONT+sorted(c for c in df.columns if c.startswith("state__") and "__is_" in c);rng=np.random.default_rng(a.seed);ar=[];pr=[];br=[]
 for s in series_names:
  for per in PERIODS:
   x=df.loc[mask(df,per),s].replace([np.inf,-np.inf],np.nan).dropna().to_numpy(float)
   if len(x)<100:continue
   ar.append(dict(series=s,period=per,n=len(x),**acf_stats(x,a.max_acf_lag)));f,p=psd(x);pp,pw=peak(f,p);nullp=[];nullb={b:[] for b in BANDS}
   for _ in range(a.surrogates):
    fs,ps=psd(perm(x,a.block_size,rng));_,sp=peak(fs,ps);nullp.append(sp)
    for b in BANDS:nullb[b].append(bpower(fs,ps,*b))
   npk=np.asarray(nullp,float);npk=npk[np.isfinite(npk)];pv=(np.sum(npk>=pw)+1)/(len(npk)+1) if len(npk) and np.isfinite(pw) else np.nan;pr.append(dict(series=s,period=per,n=len(x),peak_period_days=pp,peak_power=pw,peak_surrogate_p=pv))
   for b in BANDS:
    obs=bpower(f,p,*b);nn=np.asarray(nullb[b],float);nn=nn[np.isfinite(nn)];pv=(np.sum(nn>=obs)+1)/(len(nn)+1) if len(nn) and np.isfinite(obs) else np.nan;br.append(dict(series=s,period=per,n=len(x),band_lo_days=b[0],band_hi_days=b[1],observed_band_power=obs,surrogate_p=pv,power_to_surrogate_median=float(obs/np.median(nn)) if len(nn) and np.median(nn)>0 else np.nan))
 acf=pd.DataFrame(ar);peaks=pd.DataFrame(pr);bands=pd.DataFrame(br);sr=[]
 for s in sorted(bands.series.unique()):
  for lo,hi in BANDS:
   q=bands[(bands.series==s)&(bands.band_lo_days==lo)&(bands.band_hi_days==hi)];sig=sorted(q[(q.period!="full")&(q.surrogate_p<.05)].period.unique());sr.append(dict(series=s,band_lo_days=lo,band_hi_days=hi,n_nonoverlap_periods_significant=len(sig),significant_periods=";".join(sig),full_sample_significant=bool(((q.period=="full")&(q.surrogate_p<.05)).any()),crisis_period_only=sig==["2020_2024"],passes_stability_screen=len(sig)>=2 and sig!=["2020_2024"]))
 stab=pd.DataFrame(sr);acf.to_csv(a.outdir/"acf_summary.csv",index=False);peaks.to_csv(a.outdir/"spectral_peaks.csv",index=False);bands.to_csv(a.outdir/"band_power_tests.csv",index=False);stab.to_csv(a.outdir/"cross_period_stability.csv",index=False);(a.outdir/"run_manifest.json").write_text(json.dumps(dict(data=str(a.data),series=series_names,bands=BANDS,surrogates=a.surrogates,block_size=a.block_size,max_acf_lag=a.max_acf_lag,seed=a.seed),indent=2))
 passing=stab[stab.passes_stability_screen];lines=["# Frequency Structure Audit Report","","## Primary conclusion","",("No series/band passed the cross-period stability screen; persistence may exist without a defensible slow cycle." if passing.empty else "At least one series/band passed the cross-period stability screen. This supports reproducible low-frequency structure, not a fixed deterministic cycle."),"","## Passing rows","","```text",passing.to_csv(index=False),"```","","## Interpretation","","Large ACF timescales indicate persistence, not periodicity. Slow structure is interesting only when broad-band power exceeds block-surrogate nulls in at least two non-overlapping periods."];(a.outdir/"frequency_structure_report.md").write_text("\n".join(lines));print("\n=== Cross-period stable slow-frequency bands ===");print("None" if passing.empty else passing.to_string(index=False));print(f"\nWrote frequency audit outputs to {a.outdir}")
if __name__=="__main__":main()
