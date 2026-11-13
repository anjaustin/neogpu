import json,sys,collections
for f in sys.argv[1:]:
  print('##',f)
  for l in open(f):
    j=json.loads(l)
    if 'policy' not in j: print(l.strip()); continue
    a=j['age_us']
    print('%-3s s%d %-22s x%-4s hp%-7d ema%5.0f reg%.2f tr%-3d rep%3d/%-4.0f %.2f%% med%5.1f p90 %5.1f max%5.1f stepsd %5.2f fps%5.1f ms%s'%(
      j['kworker_rt'],j['spin'],j['pattern'] or ('N%d'%j['shader_iterations']),j['extra'],j['hold_permille'],j['present_interval_ema_us'],
      j['hold_regime_frac'],j['regime_transitions'],j['refreshes_repeated'],j['refreshes'],100*j['refreshes_repeated']/j['refreshes'],
      a['median']/1e3,a['p90']/1e3,a['max']/1e3,j['step_sd_ms'],j['fps'],j['pattern_ms']))
