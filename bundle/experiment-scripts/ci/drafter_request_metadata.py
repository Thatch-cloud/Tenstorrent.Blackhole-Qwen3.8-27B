"""Attach observed completion and audited sampler identity before request validation."""

from sampling_link_policy import SOURCES


def record_request(report, result, drafter, sampling, fabric_sources):
    links = sampling.num_argmax_gather_links
    if (drafter not in ('dspark', 'dflash2') or type(links) is not int or links != 4
            or fabric_sources != SOURCES or report.get('sampling_link_sources') != fabric_sources):
        raise ValueError('Observed four-link sampler and unchanged audited fabric sources required')
    emitted, eos = result.get('emitted'), result.get('eos_ids')
    if (not isinstance(emitted, list) or not emitted or not isinstance(eos, (list, tuple)) or not eos
            or any(type(token) is not int for token in (*emitted, *eos))):
        raise ValueError('Actual emitted tokens and configured EOS IDs required')
    result.update(comparison_drafter=drafter, ended_with_eos=emitted[-1] in eos,
                  sampler_num_links=links, fabric_sources=dict(fabric_sources))
    report['request_checks'].append(result)
