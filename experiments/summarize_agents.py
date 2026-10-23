"""Rebuild the current comparison summary from saved results, without API calls."""
import json
import statistics
from pathlib import Path


def main():
    directory=Path(__file__).resolve().parent/'agent_compare'
    voice=json.loads((directory/'voice-decision.json').read_text())
    result={'protocol':'structured-decision-v1','backends':{}}
    usage={'input':0,'output':0,'cached':0}
    for name in ('native','langchain','pydantic'):
        offline=json.loads((directory/f'offline-{name}-decision.json').read_text())
        live=json.loads((directory/f'live-{name}-decision.json').read_text())
        result['backends'][name]={
            'offline_passed':sum(r['passed'] for r in offline['rows']),
            'offline_total':len(offline['rows']),
            'warm_chat_ms':statistics.median(r['total_seconds']*1000 for r in offline['rows'] if r['case']=='chat' and r['trial']>0),
            'warm_two_settings_ms':statistics.median(r['total_seconds']*1000 for r in offline['rows'] if r['case']=='settings' and r['trial']>0),
            'resident_mib':offline['rows'][-1]['rss_current_mib'],
            'cold_first_call_seconds':offline['rows'][0]['total_seconds'],
            'live_passed':sum(r['passed'] for r in live['rows']),
            'live_total':len(live['rows']),
            'voice_first_audio_after_asr_seconds':[r['first_audio_after_asr_seconds'] for r in voice if r['backend']==name],
        }
        for row in live['rows']:
            for entry in row['metrics'].get('usage',[]):
                for key in usage:usage[key]+=entry[key]
    for row in voice:
        for entry in row['metrics'].get('usage',[]):
            for key in usage:usage[key]+=entry[key]
    result['usage_current_protocol_pilot_and_voice']=usage
    result['reserved_budget_cny_historical_and_current']=sum(json.loads((directory/'budget.json').read_text()))
    (directory/'summary-decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
