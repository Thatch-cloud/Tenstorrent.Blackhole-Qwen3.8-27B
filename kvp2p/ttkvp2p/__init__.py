"""ttkvp2p: a vLLM KV connector that spills finished KV blocks from card DRAM to an NVMe drive and restores them,
peer to peer, with no host copies.

This package is imported in the API-server process as well as in the engine (vLLM resolves the connector class there
to decide whether the hybrid KV cache manager can stay on), so importing it must have no side effects: no ttnn, no
extension module, no threads, no files, no device access. Nothing here is imported eagerly.
"""
