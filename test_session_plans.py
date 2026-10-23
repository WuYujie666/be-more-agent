import json,unittest
from unittest.mock import Mock
from session_plans import extract_plans,apply_plans

class PlanTests(unittest.TestCase):
 def extract(self,changes,messages=None,existing=None):
  return extract_plans(Mock(generate_response=lambda **kw:json.dumps({'changes':changes})),messages or [{'role':'user','content':'明天考试'}],existing or [],'2026-10-22T23:59:00+08:00')
 def change(self,**kw):
  return dict(operation='upsert',memory_id=None,text='考试',due_date='2026-10-23',evidence=[0],**kw)
 def test_unknown_id_rejected(self):
  c=self.change();c['memory_id']='invented'
  with self.assertRaises(ValueError):self.extract([c])
 def test_assistant_is_not_evidence(self):
  with self.assertRaises(ValueError):self.extract([self.change()],messages=[{'role':'assistant','content':'明天考试'}])
 def test_missing_date_rejected(self):
  c=self.change();c['due_date']=None
  with self.assertRaises(ValueError):self.extract([c])
 def test_new_cancel_does_not_write(self):
  c=self.change();c.update(operation='cancel',due_date=None)
  batch=self.extract([c]);store=Mock()
  self.assertEqual(apply_plans(store,'default',batch,[],'s','now'),[])
  store.add.assert_not_called();store.update.assert_not_called()
 def test_cancel_existing_marks_inactive(self):
  old={'id':'a','metadata':{'value':'2026-10-23','asked':False}}
  c=self.change();c.update(operation='cancel',memory_id='a',due_date=None)
  batch=self.extract([c],existing=[old]);store=Mock()
  apply_plans(store,'default',batch,[old],'s','now')
  self.assertEqual(store.update.call_args.kwargs['metadata']['status'],'cancelled')
 def test_duplicate_output_rejected(self):
  with self.assertRaises(ValueError):self.extract([self.change(),self.change()])

if __name__=='__main__':unittest.main()
