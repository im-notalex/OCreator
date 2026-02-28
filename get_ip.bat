@echo off
echo --- Network Summary ---
:: Finds lines with IPv4, Subnet, or Gateway and removes extra spaces
ipconfig | findstr /i "IPv4 Subnet Gateway"
echo -----------------------
