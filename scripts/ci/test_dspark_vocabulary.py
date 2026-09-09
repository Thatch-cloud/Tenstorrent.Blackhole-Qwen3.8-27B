from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from dspark_vocabulary import gather_logits, shared_head_logits
from test_dspark_projection import operations, tensor


class DSparkVocabularyTests(unittest.TestCase):
    def fixture(self):
        runtime = operations()
        local = tensor((1,1,32,124160))
        gathered = tensor((1,1,32,248320))
        queries = tensor((1,1,7,248320))
        wide = tensor((1,1,7,248320),'fp32')
        runtime.linear = MagicMock(return_value=local)
        runtime.experimental = SimpleNamespace(all_gather_async=MagicMock(return_value=gathered))
        runtime.Topology = SimpleNamespace(Linear='linear')
        runtime.slice = MagicMock(return_value=queries)
        runtime.typecast = MagicMock(return_value=wide)
        runtime.topk = MagicMock(side_effect=AssertionError('DSpark requires the full vocabulary'))
        runtime.to_torch = MagicMock(side_effect=AssertionError('Host staging is forbidden'))
        runtime.from_torch = MagicMock(side_effect=AssertionError('Host staging is forbidden'))
        return runtime,local,gathered,queries,wide

    def test_full_vocabulary_and_all_seven_rows_remain_on_device(self):
        runtime,local,gathered,queries,wide = self.fixture()
        mesh,collectives = SimpleNamespace(shape=(1,2)),MagicMock()
        owned = []
        with patch('dspark_vocabulary.projection_links',return_value=4):
            result = gather_logits(runtime,mesh,collectives,local,lambda value:owned.append(value) or value)
        self.assertEqual(result,dict(full_logits=gathered,base_logits=wide))
        self.assertEqual(owned,[gathered,queries,wide])
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['dim'],3)
        self.assertEqual(runtime.experimental.all_gather_async.call_args.kwargs['num_links'],4)
        runtime.slice.assert_called_once_with(gathered,(0,0,0,0),(1,1,7,248320))
        runtime.typecast.assert_called_once_with(queries,runtime.float32)
        runtime.topk.assert_not_called()
        runtime.to_torch.assert_not_called()
        runtime.from_torch.assert_not_called()

    def test_shared_head_borrows_exact_target_weights(self):
        runtime,local,gathered,queries,wide = self.fixture()
        mesh,collectives = SimpleNamespace(shape=(1,2)),MagicMock()
        weight,normalized = object(),tensor((1,1,32,5120))
        model = SimpleNamespace(num_devices=2,vocab_size=248320,_lmhead_vocab_sharded=True,
            mesh_device=mesh,lm_head_weight=weight)
        owned = []
        with patch('dspark_vocabulary.projection_links',return_value=1):
            result = shared_head_logits(runtime,model,mesh,collectives,normalized,lambda value:owned.append(value) or value)
        runtime.linear.assert_called_once_with(normalized,weight)
        self.assertIs(result['local_logits'],local)
        self.assertEqual(owned,[local,gathered,queries,wide])
        self.assertTrue(all(value is not weight and value is not normalized for value in owned))

    def test_bad_shards_reject_before_collective(self):
        for bad in (tensor((1,1,7,124160)),tensor((1,1,32,124160),'fp32'),tensor((1,1,32,248320))):
            runtime,local,gathered,queries,wide = self.fixture()
            with self.assertRaises(ValueError):
                gather_logits(runtime,SimpleNamespace(shape=(1,2)),MagicMock(),bad,lambda value:value)
            runtime.experimental.all_gather_async.assert_not_called()

    def test_wrong_target_mesh_rejects_before_head_execution(self):
        runtime,local,gathered,queries,wide = self.fixture()
        mesh = SimpleNamespace(shape=(1,2))
        model = SimpleNamespace(num_devices=2,vocab_size=248320,_lmhead_vocab_sharded=True,
            mesh_device=object(),lm_head_weight=object())
        with self.assertRaises(ValueError):
            shared_head_logits(runtime,model,mesh,MagicMock(),tensor((1,1,32,5120)),lambda value:value)
        runtime.linear.assert_not_called()


if __name__ == '__main__':
    unittest.main()
