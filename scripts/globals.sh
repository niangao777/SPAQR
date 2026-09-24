# Dataset metadata shared by the four SPAQR main-experiment scripts.
get_dataset_info() {
    local dataset_name=$1
    local dataset_subset_name=$2
    
    case $dataset_name in
        P12)
            dataset_root_path="storage/datasets/P12"
            n_variables=36
            ;;
        HumanActivity)
            dataset_root_path="storage/datasets/HumanActivity"
            n_variables=12
            ;;
        MIMIC_III)
            dataset_root_path="storage/datasets/MIMIC_III"
            n_variables=96
            ;;
        USHCN)
            dataset_root_path="storage/datasets/USHCN"
            n_variables=5
            ;;
        *)
            echo "Error: Dataset '$dataset_name' not found in scripts/globals.sh" >&2
            return 1
            ;;
    esac
    
    return 0
}
