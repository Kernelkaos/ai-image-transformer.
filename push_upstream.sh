#!/bin/bash

# Script de Sincronização e Push Upstream
# Executado no terminal do host para contornar o confinamento do Snap.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Iniciando sincronização local do repositório Git...${NC}"

# Inicializa se não estiver inicializado
if [ ! -d .git ]; then
    git init
    git branch -M main
fi

# Limpa arquivos removidos da staging area e adiciona novos
git add -A

# Realiza o commit de reestruturação
git commit -m "chore: reset and restructuring repository to professional ML architecture"

# Garante o remote correto
git remote remove origin 2>/dev/null
git remote add origin https://github.com/KernelKaos/ai-image-transformer..git

echo -e "${YELLOW}Realizando push forçado para o GitHub...${NC}"
echo -e "${YELLOW}Nota: Insira suas credenciais ou token do GitHub se solicitado.${NC}"

git push -u origin main --force

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Repositório sincronizado e enviado com sucesso ao GitHub!${NC}"
else
    echo -e "${RED}✘ Falha ao enviar alterações para o GitHub. Verifique suas credenciais.${NC}"
fi
