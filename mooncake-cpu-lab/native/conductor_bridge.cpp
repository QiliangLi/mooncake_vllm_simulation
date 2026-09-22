#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "conductor/prefixindex/hash_strategy.h"
#include "conductor/prefixindex/prefix_indexer.h"

namespace pi = mooncake::conductor::prefixindex;
struct LabIndexer {
    pi::PrefixCacheTable table;
    pi::ContextKey context;
    pi::HashProfile profile;
    std::unique_ptr<pi::HashStrategy> strategy;
    explicit LabIndexer(int block_size) : context{"lab", "text-model", "", block_size} {
        auto err = pi::ResolveHashProfile({"vllm_v1", "sha256", "0", "low64_be"}, &profile);
        if (!err.empty()) throw std::runtime_error(err);
        strategy = pi::CreateHashStrategy(profile, &err);
        if (!err.empty()) throw std::runtime_error(err);
    }
};
static thread_local std::string last_error;
extern "C" {
const char* mc_error() { return last_error.c_str(); }
void* mc_new(int block_size) {
    try { return new LabIndexer(block_size); }
    catch (const std::exception& e) { last_error=e.what(); return nullptr; }
}
void mc_delete(void* p) { delete static_cast<LabIndexer*>(p); }
int mc_register(void* p, const char* worker) {
    auto& x=*static_cast<LabIndexer*>(p);
    auto r=x.table.Register({x.context,x.profile,worker,0,x.context.block_size,0});
    last_error=r.error; return r.error.empty()?0:-1;
}
int mc_seed(void* p, const int32_t* tokens, int count, const char* object_id, int remove) {
    auto& x=*static_cast<LabIndexer*>(p);
    std::vector<pi::HashBlock> blocks;
    last_error=x.strategy->Compute(x.context,{tokens,static_cast<size_t>(count)},std::nullopt,&blocks);
    if (!last_error.empty()) return -1;
    std::vector<pi::ProjectedPrefix> prefixes;
    for (const auto& b:blocks) prefixes.push_back(b.projected);
    pi::SharedMutation m{x.context,prefixes,pi::StorageTier::kDisk,
        {"lab-pool-events","shared-pool",object_id},x.context.block_size,0};
    last_error=remove?x.table.RemoveShared(m):x.table.StoreShared(m);
    return last_error.empty()?0:-1;
}
int64_t mc_query(void* p, const int32_t* tokens, int count, const char* worker) {
    auto& x=*static_cast<LabIndexer*>(p);
    auto result=x.table.Query(x.context,{tokens,static_cast<size_t>(count)},std::nullopt,std::string(worker));
    auto it=result.find(worker);
    return it==result.end()?0:it->second.longest_match_tokens;
}
int mc_hash(void* p, const int32_t* tokens, int count, uint64_t* values) {
    auto& x=*static_cast<LabIndexer*>(p);
    std::vector<pi::HashBlock> blocks;
    last_error=x.strategy->Compute(x.context,{tokens,static_cast<size_t>(count)},std::nullopt,&blocks);
    if (!last_error.empty()) return -1;
    for(size_t i=0;i<blocks.size();++i) values[i]=blocks[i].projected.value;
    return static_cast<int>(blocks.size());
}
}
