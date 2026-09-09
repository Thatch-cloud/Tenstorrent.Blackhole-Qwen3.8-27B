from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_pipeline import INPUTS
from dspark_target import noise_embeddings, propose
from test_dspark_projection import tensor


class DSparkTargetTests(unittest.TestCase):
    def fixture(self):
        runtime = MagicMock()
        runtime.uint32,runtime.bfloat16 = 'uint32','bf16'
        runtime.ROW_MAJOR_LAYOUT,runtime.TILE_LAYOUT,runtime.DRAM_MEMORY_CONFIG = 'row','tile','dram'
        identifiers = SimpleNamespace(shape=(1,7),dtype='uint32',layout='row',memory_config=lambda:'dram')
        mesh = SimpleNamespace(shape=(1,2))
        target = SimpleNamespace(mesh_device=mesh,num_devices=2,vocab_size=248320,embd=MagicMock(),_lmhead_vocab_sharded=True)
        runtime.reshape.return_value = tensor((1,1,7,2560))
        runtime.experimental.all_gather_async.return_value = tensor((1,1,7,5120))
        runtime.pad.return_value = tensor((1,1,32,5120))
        return runtime,mesh,target,identifiers

    def test_actual_target_embedding_then_hidden_gather_and_zero_padding(self):
        runtime,mesh,target,identifiers = self.fixture()
        owned = []
        with patch('dspark_target.projection_links',return_value=4):
            result = noise_embeddings(runtime,target,mesh,MagicMock(),identifiers,lambda value:owned.append(value) or value)
        target.embd.assert_called_once_with(identifiers,memory_config='dram')
        runtime.reshape.assert_called_once_with(target.embd.return_value,(1,1,7,2560))
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['num_links'],4)
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['dim'],3)
        runtime.pad.assert_called_once_with(runtime.experimental.all_gather_async.return_value,
            [(0,0),(0,0),(0,25),(0,0)],0.0)
        self.assertIs(result,runtime.pad.return_value)
        self.assertEqual(len(owned),4)
        self.assertTrue(all(value is not identifiers for value in owned))
        runtime.to_torch.assert_not_called()
        runtime.from_torch.assert_not_called()

    def test_mismatched_target_or_wrong_query_row_count_rejects_before_embedding(self):
        for invalid in ('mesh','rows'):
            runtime,mesh,target,identifiers = self.fixture()
            if invalid=='mesh':
                target.mesh_device = object()
            else:
                identifiers.shape = (1,8)
            with self.assertRaises(ValueError):
                noise_embeddings(runtime,target,mesh,MagicMock(),identifiers,lambda value:value)
            target.embd.assert_not_called()

    def test_actual_backbone_output_feeds_borrowed_head_then_full_vocabulary_markov(self):
        runtime,mesh,target,identifiers = self.fixture()
        collectives,anchor,parameters,layers,predecessor,successor = [object() for index in range(6)]
        inputs = {name:object() for name in INPUTS if name!='noise'}
        original = dict(inputs)
        owned = []
        noise,normalized,base = object(),object(),object()
        learned = dict(backbone={'final_norm':normalized})
        logits = dict(base_logits=base)
        records = [dict(token=object()) for row in range(7)]
        with patch('dspark_target.noise_embeddings',return_value=noise), \
                patch('dspark_target.backbone',return_value=learned) as backbone, \
                patch('dspark_target.shared_head_logits',return_value=logits) as head, \
                patch('dspark_target.markov',return_value=records) as markov:
            result = propose(runtime,target,mesh,collectives,identifiers,anchor,inputs,parameters,layers,
                predecessor,successor,owned,inputs_validated=True)
        self.assertIs(backbone.call_args.args[3]['noise'],noise)
        self.assertEqual(head.call_args.args[:5],(runtime,target,mesh,collectives,normalized))
        markov.assert_called_once_with(runtime,anchor,base,predecessor,successor,owned)
        self.assertIs(result['records'][0],records[0])
        self.assertEqual(len(result['records']),7)
        self.assertEqual(inputs,original)
        self.assertEqual(runtime.mock_calls,[])

    def test_missing_explicit_contract_gate_rejects_before_any_device_call(self):
        runtime,mesh,target,identifiers = self.fixture()
        with patch('dspark_target.noise_embeddings') as embed:
            with self.assertRaises(ValueError):
                propose(runtime,target,mesh,MagicMock(),identifiers,object(),{}, {},[],object(),object(),[])
            embed.assert_not_called()


if __name__=='__main__':
    unittest.main()
