
input="mitte.json"
while IFS= read -r line
do
    echo "$line" > last.txt
done < "$input"
