Hola!

En final_preparation ya hay una version de como quedan los datos
Si se quiere generar unos nuevos hay que primero correr ORF y dsps con los resultados del script correr Folmer
Si se quiere trabajar con otro dataset del dado solo se cambia el input_csv_path en el archivo ORF.py al final

# --- Usage ---
filter_data_and_create_checkpoint(
     input_csv_path='dataset_animalia.csv', 
     output_fasta_path='coi_orf_validated_leray.fasta',
     checkpoint_csv_path='coi_class_balance_checkpoint_leray.csv'
 )