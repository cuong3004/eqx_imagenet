batch_size = 512*8
batch_size_valid = batch_size

train_set_len = 1281167 # for part 0 and for part 1: 655167
train_step_epoch = -(-train_set_len // batch_size)
# train_step_epoch = 10

valid_set_len = 50000
valid_step_epoch = -(-valid_set_len // batch_size_valid)
# valid_step_epoch = 10

args = {
    "input_dtype" : "bfloat16",
    "train_dirs" : [
        "gs://kds-c4cba4c00046f81926f20139f46645709fa1603a292f9f09205c1f5e",
        "gs://kds-479b543df23eb937562459c88992db7890d5fe810a9e77d67406f4a7"],
    "valid_dirs" : [
        "gs://kds-49298bb9b60e7369a0378148f07c8fd7c3671f304448c0a5a078dad0"
        ],
    "batch_size_train" : batch_size,
    "batch_size_valid" : batch_size_valid,
    "train_step_epoch" : train_step_epoch,
    "valid_step_epoch" : valid_step_epoch
}