-- Create indices for stackoverflow database tables to be used by ml algorithms - extend to cover extra fields for export script
use stackoverflow;
create index Posts_idx_5 on Posts(PostTypeId);
create index Posts_idx_6 on Posts(CreationDate);