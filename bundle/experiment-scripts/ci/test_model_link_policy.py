from types import SimpleNamespace
import unittest

from model_link_policy import target_links


class Collective:
    def get_num_links(self, cluster_axis=None):
        return 2


def model_fixture():
    collective = Collective()
    layers = [SimpleNamespace(tt_ccl=collective, attention=SimpleNamespace(tt_ccl=collective),
        feed_forward=SimpleNamespace(tt_ccl=collective)) for unused in range(64)]
    return SimpleNamespace(mesh_device=SimpleNamespace(shape=[1, 2]), tt_ccl=collective, layers=layers)


class ModelLinkPolicyTests(unittest.TestCase):
    def test_request_local_override_covers_all_target_owners_and_not_other_collectives(self):
        model, other = model_fixture(), Collective()
        for links in (2, 4):
            with target_links(model, links) as audit:
                for layer in model.layers:
                    for owner in (layer, layer.attention, layer.feed_forward):
                        self.assertEqual(owner.tt_ccl.get_num_links(0), links)
                self.assertEqual(model.tt_ccl.get_num_links(), links)
                self.assertEqual(model.tt_ccl.get_num_links(1), links)
                self.assertEqual(other.get_num_links(), 2)
            self.assertEqual(audit['calls'], dict(default=1, axis0=192, axis1=1))
            self.assertEqual(audit['owners_validated'], 193)
            self.assertTrue(audit['restored'])
            self.assertEqual(model.tt_ccl.get_num_links(0), 2)
            self.assertNotIn('get_num_links', vars(model.tt_ccl))
            self.assertNotIn('_qwen_target_link_scope', vars(model))
            self.assertNotIn('_qwen_target_link_scope', vars(model.tt_ccl))

    def test_exception_restores_an_existing_instance_override(self):
        model = model_fixture()
        original = lambda axis=None: 1
        model.tt_ccl.get_num_links = original
        with self.assertRaisesRegex(RuntimeError, 'failure'):
            with target_links(model, 4) as audit:
                raise RuntimeError('failure')
        self.assertIs(model.tt_ccl.get_num_links, original)
        self.assertTrue(audit['restored'])

    def test_invalid_geometry_ownership_counts_or_nested_scope_cannot_mutate(self):
        for mutation in ('links', 'boolean', 'mesh', 'layers', 'owner', 'nested', 'axis'):
            model = model_fixture()
            links = 4
            if mutation == 'links':
                links = 3
            elif mutation == 'boolean':
                links = True
            elif mutation == 'mesh':
                model.mesh_device.shape = [2, 1]
            elif mutation == 'layers':
                model.layers.pop()
            elif mutation == 'owner':
                model.layers[0].attention.tt_ccl = Collective()
            with self.subTest(mutation=mutation), self.assertRaises((ValueError, RuntimeError)):
                with target_links(model, links):
                    if mutation == 'nested':
                        with target_links(model, links):
                            self.fail('Nested scope must fail')
                    elif mutation == 'axis':
                        model.tt_ccl.get_num_links(True)
            self.assertEqual(model.tt_ccl.get_num_links(), 2)
            self.assertNotIn('get_num_links', vars(model.tt_ccl))


if __name__ == '__main__':
    unittest.main()
