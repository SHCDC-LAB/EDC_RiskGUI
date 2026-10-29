from pathlib import Path
import json, hashlib, warnings, glob, joblib, math, os
import numpy as np, pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
warnings.filterwarnings('ignore')

ROOT=Path(__file__).resolve().parents[1]
import argparse
parser=argparse.ArgumentParser(description="EDC-AOPRisk V10-B external-evidence-gated candidate predictor")
parser.add_argument("--input",required=True)
parser.add_argument("--output-dir",required=True)
parser.add_argument("--smiles-col",default="SMILES")
parser.add_argument("--name-col",default="Name")
parser.add_argument("--cas-col",default="CAS")
a=parser.parse_args()
MATRIX=ROOT/'data/model_selection_registry_v10b_final.csv'
INPUT=Path(a.input)
OUT=Path(a.output_dir);OUT.mkdir(parents=True,exist_ok=True)

# paths
XPATH=ROOT/'data/parent_features_1034.npy'
FIPATH=ROOT/'data/parent_feature_index.csv'
ACTLAB=ROOT/'data/parent_activity_labels_v9.csv'
POTUSE=ROOT/'data/parent_potency_usable_v9.csv'
POTSTRICT=ROOT/'data/parent_potency_strict_v9.csv'
ACTDIR=ROOT/'models/activity'; V10ACTDIR=ROOT/'models/activity_v10_hybrid'; POTDIR=ROOT/'models/potency'; THDIR=ROOT/'models/threshold'
DESC_COLS=pd.read_csv(ROOT/'data/v10a_rdkit_2d_descriptor_columns.csv')
DESC_RT=np.load(ROOT/'data/v10a_descriptor_runtime.npz')
DESC_FUNCS=dict(Descriptors.descList)
DESC_NAMES=DESC_COLS.descriptor_name.astype(str).tolist()
DESC_MED=DESC_RT['impute_median'].astype(np.float32); DESC_LO=DESC_RT['clip_low'].astype(np.float32); DESC_HI=DESC_RT['clip_high'].astype(np.float32)

# chemistry functions
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1024)
chooser=rdMolStandardize.LargestFragmentChooser(); uncharger=rdMolStandardize.Uncharger()
def std(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None or any(a.GetAtomicNum()==0 for a in m.GetAtoms()): return None,None,None
    try:
        m=rdMolStandardize.Cleanup(m);m=chooser.choose(m);m=uncharger.uncharge(m);Chem.SanitizeMol(m)
        cs=Chem.MolToSmiles(m,canonical=True);sc=MurckoScaffold.GetScaffoldForMol(m)
        ss=Chem.MolToSmiles(sc,canonical=True) if sc and sc.GetNumAtoms() else 'ACYCLIC_'+hashlib.sha1(cs.encode()).hexdigest()[:12]
        return cs,ss,m
    except Exception:return None,None,None

def feat(m):
    fp=gen.GetFingerprint(m);b=np.zeros(1024,np.float32);DataStructs.ConvertToNumpyArray(fp,b)
    d=np.array([Descriptors.MolWt(m),Crippen.MolLogP(m),rdMolDescriptors.CalcTPSA(m),Lipinski.NumHDonors(m),Lipinski.NumHAcceptors(m),Lipinski.NumRotatableBonds(m),m.GetNumHeavyAtoms(),rdMolDescriptors.CalcNumRings(m),rdMolDescriptors.CalcFractionCSP3(m),Chem.GetFormalCharge(m)],np.float32)
    return np.r_[b,d]

def feat2d(m):
    vals=np.full(len(DESC_NAMES),np.nan,np.float32)
    if m is not None and m.GetNumHeavyAtoms()<=60:
        for j,n in enumerate(DESC_NAMES):
            f=DESC_FUNCS.get(n)
            if f is None: continue
            try:
                v=float(f(m)); vals[j]=v if np.isfinite(v) else np.nan
            except Exception: pass
    bad=~np.isfinite(vals); vals[bad]=DESC_MED[bad]
    return np.clip(vals,DESC_LO,DESC_HI).astype(np.float32)

def activity_matrix(bundle,Xbaseq,X2q):
    inds=bundle.get('descriptor_indices',[])
    if not inds: return Xbaseq
    return np.concatenate([Xbaseq,X2q[:,np.asarray(inds,int)]],axis=1)

def v10cols(df):
    return df.rename(columns={c:c.replace('_v9','_v10') for c in df.columns})

def logit(x):
    x=np.clip(np.asarray(x,float),1e-6,1-1e-6); return np.log(x/(1-x))
def cprob(bundle,raw):
    c=bundle.get('calibrator')
    if c is None:return np.asarray(raw,float)
    inp=bundle.get('calibration_input','raw_probability')
    x=np.asarray(raw,float)
    if inp=='logit_raw_probability': x=logit(x)
    return c.predict_proba(x.reshape(-1,1))[:,1]

def adflag(nn,top5,scn,z,exact):
    if exact:return 'observed_parent_match'
    if not np.isfinite(nn) or not np.isfinite(z):return 'out_of_domain'
    if (nn>=.60 and top5>=.40 and z<=3.0) or (nn>=.50 and scn>=3 and z<=2.5):return 'in_domain'
    if (nn>=.40 and z<=4.0) or (nn>=.35 and scn>=1 and z<=3.5):return 'borderline_domain'
    return 'out_of_domain'

def make_stat(idx,Xall):
    if len(idx)==0:return {'rows':np.array([],int),'mu':np.zeros(10),'sd':np.ones(10),'sc':{}}
    rows=idx.feature_row.astype(int).to_numpy();ds=np.asarray(Xall[rows,1024:],float);mu=ds.mean(0);sd=np.where(ds.std(0)<1e-6,1,ds.std(0));return {'rows':rows,'mu':mu,'sd':sd,'sc':idx.parent_scaffold.value_counts().to_dict()}
def sim_stats(S,st,qdesc,qsc):
    rows=st['rows']
    if len(rows)==0:return np.nan,np.nan,0,np.nan
    s=S[rows];nn=float(np.nanmax(s));k=min(5,len(s));top=float(np.median(np.partition(s,len(s)-k)[-k:]));z=float(np.sqrt(np.mean(np.clip((qdesc-st['mu'])/st['sd'],-20,20)**2)));scn=int(st['sc'].get(qsc,0));return nn,top,scn,z

# metadata and model registry
matrix=pd.read_csv(MATRIX).copy()
if 'endpoint_key' not in matrix.columns:
    matrix['endpoint_key']=matrix['endpoint_key_v8_8']
if len(matrix)!=41 or matrix['endpoint_key'].nunique()!=41:
    raise RuntimeError(f'Frozen V10B registry must contain exactly 41 unique endpoints; got rows={len(matrix)}, unique={matrix["endpoint_key"].nunique()}')
gate_registry=pd.read_csv(ROOT/'data/V10B_EXTERNAL_EVIDENCE_GATE_REGISTRY.csv')
expected_upgrade=set(gate_registry.loc[gate_registry['gate_decision'].eq('UPGRADE_V10'),'endpoint'].astype(str))
registry_upgrade=set(matrix.loc[matrix['v10b1_final_activity_source'].eq('V10_UPGRADE_EXTERNAL_PASS'),'endpoint_key'].astype(str))
if expected_upgrade != registry_upgrade:
    raise RuntimeError(f'External gate/41-endpoint registry mismatch: gate={sorted(expected_upgrade)} registry={sorted(registry_upgrade)}')
# new neural result: both shadow candidates not selected; downgrade continuous role based on traditional performance
for ep in ['NIS_SLC5A5_INHIBITION','SRD5A2_INHIBITION']:
    m=matrix.endpoint_key.eq(ep)
    matrix.loc[m,'neural_status']='not_selected_after_assay_rebuild_retest'
    matrix.loc[m,'neural_challenger']='research_shadow_only'
    matrix.loc[m,'potency_role_candidate']='weak_auxiliary'
    matrix.loc[m,'notes']=matrix.loc[m,'notes'].fillna('').astype(str)+' | v9 retest: GNN advantage not robust under rebuilt labels.'

# exact bundle inventories
v9_act_files={joblib.load(f)['endpoint_key']:Path(f) for f in glob.glob(str(ACTDIR/'*.pkl'))}
v10_act_files={joblib.load(f)['endpoint_key']:Path(f) for f in glob.glob(str(V10ACTDIR/'*.pkl'))}
all_endpoints=set(matrix['endpoint_key'].astype(str))
if set(v9_act_files) != all_endpoints:
    raise RuntimeError(f'V9 activity inventory mismatch: missing={sorted(all_endpoints-set(v9_act_files))}, extra={sorted(set(v9_act_files)-all_endpoints)}')
if set(v10_act_files) != expected_upgrade:
    raise RuntimeError(f'V10 activity inventory must exactly equal externally approved upgrade set: files={sorted(v10_act_files)} gate={sorted(expected_upgrade)}')
act_files={ep:(v10_act_files[ep] if ep in expected_upgrade else v9_act_files[ep]) for ep in all_endpoints}
pot_files={joblib.load(f)['endpoint_key']:Path(f) for f in glob.glob(str(POTDIR/'*.pkl'))}
th_files={}
for f in glob.glob(str(THDIR/'*.pkl')):
    b=joblib.load(f); th_files[(b['endpoint_key'],float(b['threshold_pac50']))]=Path(f)

# feature and parent index
Xall=np.load(XPATH,mmap_mode='r');fi=pd.read_csv(FIPATH)
fi_map=fi.set_index('parent_smiles')

def attach_scaf(d, ep_col):
    x=d.copy(); x['endpoint_key']=x[ep_col].astype(str)
    x=x.merge(fi[['parent_smiles','parent_scaffold','feature_row']],on=['parent_smiles','feature_row'],how='left')
    return x
act=attach_scaf(pd.read_csv(ACTLAB),'ep')
use=attach_scaf(pd.read_csv(POTUSE),'endpoint_key')
strict=attach_scaf(pd.read_csv(POTSTRICT),'ep')

# actual model training potency index and broader observed usable index
pot_train_parts=[]
for ep,path in pot_files.items():
    b=joblib.load(path);src=b.get('potency_label_source','')
    if 'strict' in src.lower(): d=strict[strict.endpoint_key.eq(ep)].copy()
    else: d=use[use.endpoint_key.eq(ep)].copy()
    d['model_training_source']=src;pot_train_parts.append(d)
pottrain=pd.concat(pot_train_parts,ignore_index=True) if pot_train_parts else pd.DataFrame()

# activity and potency endpoint caches
Ac={ep:g.drop_duplicates('parent_smiles').copy() for ep,g in act.groupby('endpoint_key',sort=False)}
Pc={ep:g.drop_duplicates('parent_smiles').copy() for ep,g in pottrain.groupby('endpoint_key',sort=False)}
ObsP={ep:g.drop_duplicates('parent_smiles').copy() for ep,g in use.groupby('endpoint_key',sort=False)}
Ast={ep:make_stat(g,Xall) for ep,g in Ac.items()};Pst={ep:make_stat(g,Xall) for ep,g in Pc.items()}

# input
inp=pd.read_csv(INPUT)
if a.smiles_col not in inp.columns:
    raise SystemExit(f'Missing SMILES column: {a.smiles_col}')
if 'input_row_id' not in inp.columns:
    inp.insert(0,'input_row_id',np.arange(1,len(inp)+1))
if a.name_col not in inp.columns:
    inp[a.name_col]=inp['input_row_id'].map(lambda x:f'chemical_{x}')
if a.cas_col not in inp.columns:
    inp[a.cas_col]=''
if a.name_col!='Name' or a.cas_col!='CAS':
    inp=inp.rename(columns={a.name_col:'Name',a.cas_col:'CAS'})
sv=[std(x) for x in inp[a.smiles_col]]
inp['parent_smiles_v9']=[x[0] for x in sv];inp['parent_scaffold_v9']=[x[1] for x in sv]
F=[feat(x[2]) if x[2] is not None else None for x in sv]
D2=[feat2d(x[2]) if x[2] is not None else None for x in sv]
valid=np.array([i for i,x in enumerate(F) if x is not None],int);Xq=np.vstack([F[i] for i in valid]) if len(valid) else np.empty((0,1034),np.float32)
X2q=np.vstack([D2[i] for i in valid]) if len(valid) else np.empty((0,len(DESC_NAMES)),np.float32)
valid_pos={idx:j for j,idx in enumerate(valid)}
# global Tanimoto to frozen feature space
Tb=np.asarray(Xall[:,:1024]>0.5,dtype=np.float64);ts=Tb.sum(1)
SIM=np.full((len(inp),len(Tb)),np.nan,np.float64)
if len(valid):
    Q=np.asarray(Xq[:,:1024]>0.5,np.float64);inter=Q@Tb.T;den=Q.sum(1)[:,None]+ts[None,:]-inter;SIM[valid]=np.divide(inter,den,out=np.zeros_like(inter),where=den>0)

# scoring
rows=[]
for _,meta in matrix.sort_values(['endpoint_relevance_rank','endpoint_key']).iterrows():
    ep=meta.endpoint_key;rank=int(meta.endpoint_relevance_rank);core=rank==1
    ab=joblib.load(act_files[ep]);raw=np.full(len(inp),np.nan);cal=np.full(len(inp),np.nan)
    if len(valid):
        Xa=activity_matrix(ab,Xq,X2q); r=ab['model'].predict_proba(Xa)[:,1];raw[valid]=r;cal[valid]=cprob(ab,r)
    pb=joblib.load(pot_files[ep]) if ep in pot_files else None
    pac=np.full(len(inp),np.nan)
    if pb is not None and len(valid):pac[valid]=pb['model'].predict(Xq)
    q80=float(pb.get('q_abs_80',np.nan)) if pb else np.nan;q90=float(pb.get('q_abs_90',np.nan)) if pb else np.nan
    ql80=float(pb.get('q_lower_80',np.nan)) if pb else np.nan;ql90=float(pb.get('q_lower_90',np.nan)) if pb else np.nan
    # thresholds
    tv={}
    for th in (5.5,6.0):
        rr=np.full(len(inp),np.nan);cc=np.full(len(inp),np.nan);status='not_trained'
        if (ep,th) in th_files:
            tb=joblib.load(th_files[(ep,th)]);status=tb.get('deployment_status','trained')
            if len(valid):
                r=tb['model'].predict_proba(Xq)[:,1];rr[valid]=r;cc[valid]=cprob(tb,r)
        tv[th]=(rr,cc,status)
    # maps/stats
    ai=Ac.get(ep,pd.DataFrame());pi=Pc.get(ep,pd.DataFrame());oi=ObsP.get(ep,pd.DataFrame())
    amap=ai.set_index('parent_smiles') if len(ai) else pd.DataFrame()
    pmap=pi.set_index('parent_smiles') if len(pi) else pd.DataFrame()
    omap=oi.set_index('parent_smiles') if len(oi) else pd.DataFrame()
    ast=Ast.get(ep,make_stat(pd.DataFrame(columns=['feature_row','parent_scaffold']),Xall));pst=Pst.get(ep,make_stat(pd.DataFrame(columns=['feature_row','parent_scaffold']),Xall))
    rel=str(meta.potency_role_candidate)
    # five expression redefined endpoints are supporting only and no direct tier effect
    expression_redefined=str(meta.curation_status)=='EXTRACT_FOR_ONTOLOGY_REDEFINITION_ONLY'
    for i,z in inp.iterrows():
        parent=z.parent_smiles_v9;sc=z.parent_scaffold_v9
        oa=np.nan;op=np.nan;on=np.nan;ostrict=False
        if isinstance(parent,str) and len(ai) and parent in amap.index:oa=float(amap.loc[parent].activity)
        if isinstance(parent,str) and len(oi) and parent in omap.index:
            op=float(omap.loc[parent].p_ac50);on=float(omap.loc[parent].w) if 'w' in omap.columns else np.nan
            ostrict=bool(len(strict[(strict.endpoint_key.eq(ep)) & (strict.parent_smiles.eq(parent))]))
        if i in valid_pos:
            qi=valid_pos[i];qd=Xq[qi,1024:]
            ann,atop,asc,az=sim_stats(SIM[i],ast,qd,sc)
            pnn,ptop,psc,pz=sim_stats(SIM[i],pst,qd,sc) if pb is not None else (np.nan,np.nan,0,np.nan)
            aad=adflag(ann,atop,asc,az,pd.notna(oa));pad=adflag(pnn,ptop,psc,pz,pd.notna(op)) if pb is not None else 'not_available'
        else: ann=pnn=np.nan;atop=ptop=az=pz=np.nan;asc=psc=0;aad=pad='invalid'
        if pd.notna(op):joint='observed_parent_match';origin='Observed_v4_3_parent_potency_evidence'
        elif pb is None:joint=aad;origin='Activity_model_only'
        elif aad=='out_of_domain' or pad=='out_of_domain':joint='out_of_domain';origin='Model_extrapolation'
        elif aad=='in_domain' and pad=='in_domain':joint='in_domain';origin='Model_interpolation'
        else:joint='borderline_domain';origin='Model_borderline_interpolation'
        p55=tv[5.5][1][i];p60=tv[6.0][1][i];s55=cal[i]*p55 if np.isfinite(p55) else np.nan;s60=cal[i]*p60 if np.isfinite(p60) else np.nan
        lo80=pac[i]-q80 if np.isfinite(pac[i]) and np.isfinite(q80) else np.nan;hi80=pac[i]+q80 if np.isfinite(pac[i]) and np.isfinite(q80) else np.nan
        lo90=pac[i]-q90 if np.isfinite(pac[i]) and np.isfinite(q90) else np.nan;hi90=pac[i]+q90 if np.isfinite(pac[i]) and np.isfinite(q90) else np.nan
        l80=pac[i]-ql80 if np.isfinite(pac[i]) and np.isfinite(ql80) else np.nan;l90=pac[i]-ql90 if np.isfinite(pac[i]) and np.isfinite(ql90) else np.nan
        qual=rel in ['primary_quantitative','ranking_support']
        if core:
            c55=tv[5.5][2]=='deployable' and cal[i]>=.60 and np.isfinite(p55) and p55>=.60 and s55>=.35
            c60=tv[6.0][2]=='deployable' and cal[i]>=.65 and np.isfinite(p60) and p60>=.65 and s60>=.45
            q55=qual and np.isfinite(l80) and l80>=5.5;q60=qual and np.isfinite(l90) and l90>=6.0
            if pd.notna(op):
                conc='observed_potency_evidence';sig='Observed high-potency Core-EATS evidence' if op>=6 else 'Observed medium-potency Core-EATS evidence' if op>=5.5 else 'Observed weak-potency Core-EATS evidence' if op>=4.5 else 'Observed low-potency Core-EATS evidence'
            elif joint=='out_of_domain':conc='out_of_domain_no_confident_potency_claim';sig='Uncertain | out-of-domain Core-EATS model'
            elif c60 and q60:conc='high_potency_concordant_classifier_plus_continuous';sig='Model high-potency | concordant confidence-qualified'
            elif c60:conc='high_classifier_supported_continuous_uncertain';sig='Model high classifier signal | continuous potency uncertain'
            elif q60:conc='high_continuous_supported_classifier_not_concordant';sig='Model high continuous signal | classifier not concordant'
            elif c55 and q55:conc='medium_potency_concordant_classifier_plus_continuous';sig='Model medium-potency | concordant confidence-qualified'
            elif c55:conc='medium_classifier_supported_continuous_uncertain';sig='Model medium classifier signal | continuous potency uncertain'
            elif q55:conc='medium_continuous_supported_classifier_not_concordant';sig='Model medium continuous signal | classifier not concordant'
            elif cal[i]>=.65:conc='no_confidence_qualified_potency_support';sig='Core-EATS activity signal | potency not confidence-qualified'
            else:conc='no_confidence_qualified_potency_support';sig='Low/insufficient Core-EATS model evidence'
            grade='A_strict_observed' if pd.notna(op) and ostrict else 'B_observed_usable' if pd.notna(op) else 'D' if joint=='out_of_domain' else 'B' if 'concordant confidence-qualified' in sig else 'C_evidence_disagreement' if ('classifier signal' in sig or 'continuous signal' in sig) else 'C'
            role='core_eats_primary_hazard';tier=True
        else:
            dep55=tv[5.5][2]=='supporting_signal_deployable';dep60=tv[6.0][2]=='supporting_signal_deployable'
            c55=dep55 and cal[i]>=.60 and np.isfinite(p55) and p55>=.60;c60=dep60 and cal[i]>=.65 and np.isfinite(p60) and p60>=.65
            q55=qual and np.isfinite(l80) and l80>=5.5;q60=qual and np.isfinite(l90) and l90>=6
            if pd.notna(op):conc='observed_supporting_potency_evidence';sig='Observed high-potency supporting evidence' if op>=6 else 'Observed medium-potency supporting evidence' if op>=5.5 else 'Observed weak-potency supporting evidence' if op>=4.5 else 'Observed low-potency supporting evidence'
            elif joint=='out_of_domain':conc='out_of_domain_no_confident_supporting_potency_claim';sig='Uncertain supporting signal | out-of-domain'
            elif c60 and q60:conc='high_supporting_potency_concordant_classifier_plus_continuous';sig='Model strong supporting signal | concordant classifier + continuous'
            elif c60:conc='high_supporting_classifier_only_continuous_uncertain';sig='Model strong supporting classifier signal | continuous uncertain'
            elif c55 and q55:conc='medium_supporting_potency_concordant_classifier_plus_continuous';sig='Model moderate supporting signal | concordant classifier + continuous'
            elif c55:conc='medium_supporting_classifier_only_continuous_uncertain';sig='Model moderate supporting classifier signal | continuous uncertain'
            elif q60:conc='high_supporting_continuous_only_classifier_not_concordant';sig='Model strong continuous supporting signal | classifier not concordant'
            elif q55:conc='medium_supporting_continuous_only_classifier_not_concordant';sig='Model moderate continuous supporting signal | classifier not concordant'
            elif cal[i]>=.65:conc='no_confidence_qualified_supporting_potency';sig='Model activity supporting signal | potency uncertain'
            else:conc='no_confidence_qualified_supporting_potency';sig='Low/insufficient supporting evidence'
            grade='supporting_only';role='endocrine_adjacent_mechanistic_support' if rank==2 else 'metabolism_disposition_support';tier=False
            if expression_redefined: sig='Expression-change auxiliary evidence | '+sig; tier=False
        rows.append({
            'input_row_id':z.input_row_id,'Name':z.Name,'CAS':z.CAS,'parent_smiles_v9':parent,'endpoint_key':ep,'endpoint_relevance_rank':rank,'endpoint_role':role,'curation_status':meta.curation_status,
            'evidence_origin_v9':origin,'observed_parent_activity_label_v9':oa,'observed_parent_pAC50_v9':op,'observed_parent_n_records_v9':on,'observed_strict_confirmation_v9':ostrict,
            'activity_score_raw_v9':raw[i],'activity_probability_calibrated_v9':cal[i],'predicted_pAC50_v9':pac[i],'pAC50_reliability_v9':rel,
            'pAC50_interval80_lower_v9':lo80,'pAC50_interval80_upper_v9':hi80,'pAC50_interval90_lower_v9':lo90,'pAC50_interval90_upper_v9':hi90,'pAC50_one_sided_lower80_v9':l80,'pAC50_one_sided_lower90_v9':l90,
            'p_ge5p5_calibrated_v9':p55,'threshold_ge5p5_status_v9':tv[5.5][2],'p_ge6p0_calibrated_v9':p60,'threshold_ge6p0_status_v9':tv[6.0][2],
            'hurdle_activity_x_ge5p5_v9':s55,'hurdle_activity_x_ge6p0_v9':s60,'joint_AD_v9':joint,'nearest_activity_parent_similarity_v9':ann,'nearest_potency_parent_similarity_v9':pnn,
            'potency_evidence_concordance_v9':conc,'endpoint_signal_v9':sig,'evidence_confidence_grade_v9':grade,'tier_driver_allowed_v9':tier
        })

E=pd.DataFrame(rows)

# -----------------------------------------------------------------------------
# Observed evidence QC layering: strict-confirmed vs usable-only
# -----------------------------------------------------------------------------
strict_obs = strict[['endpoint_key','parent_smiles','p_ac50','w']].drop_duplicates(['endpoint_key','parent_smiles']).rename(
    columns={'p_ac50':'observed_parent_pAC50_strict_v9','w':'observed_parent_strict_n_records_v9'})
E=E.merge(strict_obs,left_on=['endpoint_key','parent_smiles_v9'],right_on=['endpoint_key','parent_smiles'],how='left').drop(columns=['parent_smiles'],errors='ignore')
E['observed_strict_confirmation_v9']=E['observed_parent_pAC50_strict_v9'].notna()
obs=E['observed_parent_pAC50_v9'].notna(); core=E.endpoint_relevance_rank.eq(1); sup=~core
strictm=E['observed_strict_confirmation_v9']; op=E['observed_parent_pAC50_v9']; sp=E['observed_parent_pAC50_strict_v9']
E.loc[obs & strictm,'evidence_origin_v9']='Observed_v4_3_strict_parent_potency_evidence'
E.loc[obs & ~strictm,'evidence_origin_v9']='Observed_v4_3_usable_only_parent_potency_evidence'
# Core strict observed
m=core & obs & strictm
E.loc[m & sp.ge(6),'endpoint_signal_v9']='Observed QC-confirmed high-potency Core-EATS evidence'
E.loc[m & sp.ge(5.5) & sp.lt(6),'endpoint_signal_v9']='Observed QC-confirmed medium-potency Core-EATS evidence'
E.loc[m & sp.ge(4.5) & sp.lt(5.5),'endpoint_signal_v9']='Observed QC-confirmed weak-potency Core-EATS evidence'
E.loc[m & sp.lt(4.5),'endpoint_signal_v9']='Observed QC-confirmed low-potency Core-EATS evidence'
E.loc[m,'evidence_confidence_grade_v9']='A_strict_observed'
# Core usable-only observed
m=core & obs & ~strictm
E.loc[m & op.ge(9),'endpoint_signal_v9']='Observed extreme potency record | strict QC not confirmed'
E.loc[m & op.ge(6) & op.lt(9),'endpoint_signal_v9']='Observed usable-only high-potency record | strict QC not confirmed'
E.loc[m & op.ge(5.5) & op.lt(6),'endpoint_signal_v9']='Observed usable-only medium-potency record | strict QC not confirmed'
E.loc[m & op.ge(4.5) & op.lt(5.5),'endpoint_signal_v9']='Observed usable-only weak-potency Core-EATS evidence'
E.loc[m & op.lt(4.5),'endpoint_signal_v9']='Observed usable-only low-potency Core-EATS evidence'
E.loc[m & op.ge(9),'evidence_confidence_grade_v9']='C_verify_extreme_unconfirmed'
E.loc[m & op.lt(9),'evidence_confidence_grade_v9']='B_usable_observed_unconfirmed'
# Supporting observed
m=sup & obs & strictm
E.loc[m & sp.ge(6),'endpoint_signal_v9']='Observed QC-confirmed high-potency supporting evidence'
E.loc[m & sp.ge(5.5) & sp.lt(6),'endpoint_signal_v9']='Observed QC-confirmed medium-potency supporting evidence'
E.loc[m & sp.ge(4.5) & sp.lt(5.5),'endpoint_signal_v9']='Observed QC-confirmed weak-potency supporting evidence'
E.loc[m & sp.lt(4.5),'endpoint_signal_v9']='Observed QC-confirmed low-potency supporting evidence'
m=sup & obs & ~strictm
vals=[]
for _,rr in E.loc[m].iterrows():
    pref='Expression-change auxiliary evidence | ' if rr.curation_status=='EXTRACT_FOR_ONTOLOGY_REDEFINITION_ONLY' else ''
    v=rr.observed_parent_pAC50_v9
    if v>=9:s='Observed extreme supporting potency record | strict QC not confirmed'
    elif v>=6:s='Observed usable-only high-potency supporting record | strict QC not confirmed'
    elif v>=5.5:s='Observed usable-only medium-potency supporting record | strict QC not confirmed'
    elif v>=4.5:s='Observed usable-only weak-potency supporting evidence'
    else:s='Observed usable-only low-potency supporting evidence'
    vals.append(pref+s)
E.loc[m,'endpoint_signal_v9']=vals
E.loc[sup & obs,'evidence_confidence_grade_v9']='supporting_only'

# Endpoint-level output
v10cols(E).to_csv(OUT/'endpoint_level_full41_v10b_external_gated.csv',index=False)
v10cols(E[core]).to_csv(OUT/'endpoint_level_core17_v10b_external_gated.csv',index=False)
v10cols(E[sup]).to_csv(OUT/'endpoint_level_support24_v10b_external_gated.csv',index=False)

# Chemical-level summaries. Core is authoritative; Supporting is advisory.
crank={
'Observed QC-confirmed high-potency Core-EATS evidence':13,
'Observed QC-confirmed medium-potency Core-EATS evidence':12,
'Model high-potency | concordant confidence-qualified':11,
'Model high classifier signal | continuous potency uncertain':10,
'Model high continuous signal | classifier not concordant':9,
'Model medium-potency | concordant confidence-qualified':8,
'Observed usable-only high-potency record | strict QC not confirmed':7.5,
'Model medium classifier signal | continuous potency uncertain':7,
'Observed extreme potency record | strict QC not confirmed':6.5,
'Model medium continuous signal | classifier not concordant':6,
'Observed usable-only medium-potency record | strict QC not confirmed':5.5,
'Observed QC-confirmed weak-potency Core-EATS evidence':5,
'Observed usable-only weak-potency Core-EATS evidence':4,
'Core-EATS activity signal | potency not confidence-qualified':3,
'Observed QC-confirmed low-potency Core-EATS evidence':2,
'Observed usable-only low-potency Core-EATS evidence':1,
'Low/insufficient Core-EATS model evidence':0,
'Uncertain | out-of-domain Core-EATS model':-1}
def support_rank(s):
    if isinstance(s,str) and s.startswith('Expression-change auxiliary evidence | '):s=s.split(' | ',1)[1]
    if not isinstance(s,str):return -2
    if 'QC-confirmed high-potency supporting' in s:return 10
    if 'QC-confirmed medium-potency supporting' in s:return 9
    if 'strong supporting signal | concordant' in s:return 8
    if 'usable-only high-potency supporting' in s:return 7
    if 'extreme supporting potency' in s:return 6.5
    if 'QC-confirmed weak-potency supporting' in s:return 6
    if 'usable-only medium-potency supporting' in s:return 5.5
    if 'moderate supporting signal | concordant' in s:return 5
    if 'weak-potency supporting' in s:return 4
    if 'activity supporting signal' in s:return 3
    if 'low-potency supporting' in s:return 2
    if 'classifier' in s:return 4.5
    if 'continuous' in s:return 1
    if 'out-of-domain' in s:return -1
    if 'Low/insufficient' in s:return 0
    return 0
summary=[]
for rid,g in E.groupby('input_row_id',sort=False):
    base={'input_row_id':rid,'Name':g.iloc[0].Name,'CAS':g.iloc[0].CAS}
    cg=g[g.endpoint_relevance_rank.eq(1)].copy();cg['_r']=cg.endpoint_signal_v9.map(crank).fillna(-2)
    c=cg.sort_values(['_r','pAC50_one_sided_lower90_v9','hurdle_activity_x_ge6p0_v9'],ascending=False).iloc[0]
    base.update(core_top_endpoint_v9=c.endpoint_key,core_priority_v9=c.endpoint_signal_v9,core_potency_concordance_v9=c.potency_evidence_concordance_v9,core_activity_probability_v9=c.activity_probability_calibrated_v9,core_predicted_pAC50_v9=c.predicted_pAC50_v9,core_observed_pAC50_usable_v9=c.observed_parent_pAC50_v9,core_observed_pAC50_strict_v9=c.observed_parent_pAC50_strict_v9,core_AD_v9=c.joint_AD_v9,core_evidence_origin_v9=c.evidence_origin_v9,core_confidence_grade_v9=c.evidence_confidence_grade_v9)
    for tier,label in [(2,'tier2'),(3,'tier3')]:
        sg=g[g.endpoint_relevance_rank.eq(tier)].copy();sg['_r']=sg.endpoint_signal_v9.map(support_rank)
        s=sg.sort_values(['_r','activity_probability_calibrated_v9'],ascending=False).iloc[0]
        base.update({f'{label}_top_endpoint_v9':s.endpoint_key,f'{label}_top_signal_v9':s.endpoint_signal_v9,f'{label}_potency_concordance_v9':s.potency_evidence_concordance_v9,f'{label}_activity_probability_v9':s.activity_probability_calibrated_v9,f'{label}_predicted_pAC50_v9':s.predicted_pAC50_v9,f'{label}_AD_v9':s.joint_AD_v9})
    summary.append(base)
S=pd.DataFrame(summary);v10cols(S).to_csv(OUT/'chemical_summary_full41_v10b_external_gated.csv',index=False)
meta={'version':'v10B.1-audit-frozen-publication','status':'audit_frozen_publication_research_model','n_chemicals':len(S),'n_endpoint_rows':len(E),'n_activity_models':len(act_files),'n_potency_models':len(pot_files),'n_threshold_models':len(th_files),'core_priority_distribution_v10':S.core_priority_v9.value_counts().to_dict(),'activity_upgrade_endpoints':len(expected_upgrade),'potency_threshold_layers':'inherited_unchanged_from_v9','neural_decision':'NIS and SRD5A D-MPNN not selected after assay-rebuilt retest; traditional endpoint models retained; neural kept research-shadow only.','observed_evidence_policy':'Strict-confirmed observed potency can define QC-confirmed High/Medium; usable-only observed potency remains unconfirmed/caution evidence.','caveats':['H295R directional activity models use direction-specific Level-6 strict QC; analytical QC remains a secondary evidence field because exact H295R sample-level analytical-QC coverage is sparse.','Bioactivity labels are invitroDB v4.3; molecular structures come from a separately versioned CompTox/DSSTox chemical-list export joined by DTXSID. The only unmatched v4.3 assay substance is Tannins (DTXSID3021302), excluded because it has no defensible single molecular structure.']}
json.dump(meta,open(OUT/'v10b_external_gated_run_summary.json','w'),ensure_ascii=False,indent=2)
print(json.dumps(meta,ensure_ascii=False,indent=2))
