# LMM: Per-Node Statistics (SYNTHESIZE)

## P0 Spec: High-Water Marks for Channel Queues

### Changes Required

1. **hs_core.h**: Add high-water tracking
```c
// In HSSystem, after submit[CHAN_COUNT]
atomic_uint submit_hw[CHAN_COUNT];
```

2. **hs_core.c**: Update on enqueue
```c
// In hs_submit(), after atomic_fetch_add(&q->enqueue_pos):
u32 depth = atomic_load_explicit(&q->enqueue_pos, memory_order_relaxed) 
          - atomic_load_explicit(&q->dequeue_pos, memory_order_relaxed);
u32 old_hw = atomic_load_explicit(&sys->submit_hw[ch], memory_order_relaxed);
while (depth > old_hw && !atomic_compare_exchange_weak(&sys->submit_hw[ch], &old_hw, depth)) { }
```

3. **hs_core.c**: Initialize to 0 in hs_init()

4. **hs_nodes.c**: Include in OP_QUERY_FABRIC result

### Acceptance Criteria
- [ ] submit_hw starts at 0
- [ ] Under load, submit_hw increases to reflect peak
- [ ] OP_QUERY_FABRIC returns submit_hw values
- [ ] All existing tests pass
