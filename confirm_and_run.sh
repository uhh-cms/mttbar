# Function to prompt the user for confirmation before skipping a command.
# If the user types 'n' or 'no', the command is executed.
# If the user types anything else or nothing, the command is skipped.
# The user can also extend the command after typing 'n' or 'no'.

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

confirm_and_run() {
    cmd="$1"
    echo "________________________________________________________"
    echo -e "${YELLOW}$cmd${NC}"
    echo -e -n "    ${RED}Skip?${NC} (y/n): "
    read response
    case "$response" in
        [nN][oO]|[nN])
            echo -e "    ${RED}Any additional parameters?${NC} ('-ps' -> '--print-status 2,0', '-ro' -> '--remove-output 0,a,y' predefined, others possible)"
            read -e -i "    " extra_params

            # Add another case statement to handle the expansion
            case "$extra_params" in
                -ps)
                    extra_params="--print-status 2,0"
                    echo -e "    ${GREEN}Printing status...${NC}"
                    ;;
                -ro)
                    extra_params="--remove-output 0,a,y"
                    echo -e "    ${RED}Removing output...${NC}"
                    ;;
            esac
            echo -e "    ${GREEN}Running...${NC}"
            eval "$cmd $extra_params"
            ;;
        *)
            echo -e "    ${GREEN}Skipped!${NC}"
            ;;
    esac
}

no_confirm() {
    cmd="$1"
    echo "________________________________________________________"
    echo -e "${YELLOW}$cmd${NC}"
    echo -e "    ${GREEN}Running...${NC}"
    eval "$cmd"
}