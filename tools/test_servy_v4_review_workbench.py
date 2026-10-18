"""Offline synthetic tests for the manual review workbench; no Tk window opened."""
import json
import tempfile
import unittest
from pathlib import Path
from servy_v4_review_workbench import ReviewSession, NAMES, digest, validate_destination


class ReviewWorkbenchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name) / 'private_training_data'
        self.bundle = root / 'review_bundle_original'
        self.bundle.mkdir(parents=True)
        self.corpus = root / 'corpus.jsonl'
        self.corpus.write_text(''.join(json.dumps(x) + '\n' for x in [
            {'tenant_id':'a','document_id':'d1','chunk_id':'p','passage':'Real compressor passage'},
            {'tenant_id':'a','document_id':'d2','chunk_id':'n','passage':'Unrelated sensor instructions'},
        ]), encoding='utf-8')
        self.docs = [
            {'tenant_id':'a','document_id':doc,'document_versions':['1'],'source_document_sha256':[sha],
             'decision':'PENDING','reviewer_id':None,'evidence_reference':None,
             'verified_canonical_source_family':None,'approved_source_version':None}
            for doc,sha in [('d1','a'*64),('d2','b'*64)]
        ]
        self.refs = [{'split':'train','query_id':'q1','query_text':'How to repair compressor?',
                      'positive_decision':'PENDING','negative_decision':'PENDING',
                      'references':[
                          {'role':'positive','chunk_id':'p','document_id':'d1','passage':'Real compressor passage',
                           'decision':'PENDING','reviewer_id':None,'review_evidence_reference':None},
                          {'role':'negative','chunk_id':'n','document_id':'d2','passage':'Unrelated sensor instructions',
                           'decision':'PENDING','reviewer_id':None,'review_evidence_reference':None},
                      ]}]
        for name, rows in ((NAMES[0],[{'query_id':'q1'}]),(NAMES[1],[{'query_id':'q2'}]),
                           (NAMES[2],self.docs),(NAMES[3],self.refs)):
            (self.bundle/name).write_text(''.join(json.dumps(x)+'\n' for x in rows),encoding='utf-8')
        self.output = root / 'adjudication'
        self.session = ReviewSession.create(self.bundle,self.corpus,self.output)

    def test_creates_distinct_private_copy_without_original_modification(self):
        original = digest(self.bundle/NAMES[3])
        self.assertEqual(self.session.summary()['sources_pending'], 2)
        self.assertEqual(self.session.summary()['references_pending'], 2)
        self.assertEqual(original,digest(self.bundle/NAMES[3]))
        with self.assertRaises(FileExistsError):
            ReviewSession.create(self.bundle,self.corpus,self.output)

    def test_requires_explicit_evidence_and_reviewer_before_approval(self):
        with self.assertRaisesRegex(ValueError,'evidence required'):
            self.session.review_reference(0,0,decision='approved',reviewer='r',evidence='')
        with self.assertRaisesRegex(ValueError,'version differs'):
            self.session.review_source(0,decision='approved',reviewer='r',evidence='source',family='d1',version='bad')
        self.assertEqual(self.session.summary()['references_approved'],0)
        self.assertEqual(self.session.summary()['sources_approved'],0)

    def test_review_one_passage_does_not_auto_approve_other(self):
        self.session.review_reference(0,0,decision='approved',reviewer='r',evidence='source-page-1')
        self.assertEqual(self.session.summary()['references_approved'],1)
        self.assertEqual(self.session.summary()['references_pending'],1)
        self.assertEqual(self.session.labels[0]['negative_decision'],'PENDING')
        self.assertEqual(self.session.labels[0]['positive_decision'],'approved')
        self.assertEqual(self.session.summary()['label_rows_fully_approved'],0)
        self.assertEqual(digest(self.bundle/NAMES[3]),self.session.state['label_queue_original_sha256'])

    def test_review_can_resume_after_reopen_without_losing_decisions(self):
        self.session.review_source(0,decision='approved',reviewer='r',evidence='original-file-checksum-verified',family='manual-one',version='1')
        self.session.review_reference(0,0,decision='approved',reviewer='r',evidence='source-page-1')
        reopened=ReviewSession(self.output,self.corpus)
        self.assertEqual(reopened.summary()['sources_approved'],1)
        self.assertEqual(reopened.summary()['references_approved'],1)
        reopened.review_reference(0,1,decision='rejected',reviewer='r',evidence='not actually irrelevant')
        self.assertEqual(reopened.summary()['references_rejected'],1)
        self.assertEqual(reopened.labels[0]['negative_decision'],'rejected')

    def test_refuses_public_git_destination(self):
        with self.assertRaisesRegex(ValueError,'private_training_data'):
            validate_destination(self.output.parent.parent/'public-reviews',self.bundle)

    def test_changed_corpus_fails_closed(self):
        self.corpus.write_text(self.corpus.read_text()+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'corpus changed'):
            ReviewSession(self.output,self.corpus)



    def test_review_actions_remain_visible_above_expanding_viewers(self):
        # Static geometry contract: the user-facing buttons and evidence
        # inputs must appear before the scrollable panel consumes remaining
        # height. Do not create a Tk window on headless CI.
        import ast
        source = Path(__file__).with_name("servy_v4_review_workbench.py").read_text(
            encoding="utf-8")
        ast.parse(source)
        gui = source.split("def gui(session):", 1)[1].split("\\ndef main():", 1)[0]
        self.assertIn('source_actions.pack(fill="x", pady=4)', gui)
        self.assertIn('label_actions.pack(fill="x", pady=4)', gui)
        self.assertIn('ttk.Button(source_actions, text="Record APPROVED source"', gui)
        self.assertIn('ttk.Button(label_actions, text="Record APPROVED passage"', gui)
        self.assertLess(gui.index("source_actions.pack"), gui.index("source_text.pack"))
        self.assertLess(gui.index("label_actions.pack"), gui.index("label_text.pack"))
        self.assertLess(
            gui.index('ttk.Entry(sources_tab, textvariable=variable'),
            gui.index('source_text.pack'))
        self.assertLess(
            gui.index('ttk.Entry(labels_tab, textvariable=label_evidence'),
            gui.index('label_text.pack'))

if __name__=='__main__':
    unittest.main()