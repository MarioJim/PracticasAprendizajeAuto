# Dataset GENERO
genero = read.table("genero.txt", header=TRUE, sep=",")

paste("Resumen estadístico de GENERO")
summary(genero)

paste("Generando boxplot para GENERO")
png(file="generoBoxplot.png")
par(mfcol=c(1,2))
boxplot(genero$Height, ylab="inches", xlab="Height", main="Height distribution")
boxplot(genero$Weight, ylab="pounds", xlab="Weight", main="Weight distribution")
dev.off()

paste("Generando gráfica de dispersión para GENERO")
png(file="generoDispersion.png")
plot(x=genero$Weight, y=genero$Height, xlab="Weight (lbs)", ylab="Height (in)",
    main="GENERO - Weight vs Height")
dev.off()

cat("\n\n\n")

# Dataset MTCARS
mtcars = read.table("mtcars.txt", header=TRUE)

paste("Resumen estadístico de MTCARS")
summary(mtcars)

paste("Generando boxplot para MTCARS")
png(file="mtcarsBoxplot.png")
par(mfcol=c(1,3))
boxplot(mtcars$disp, xlab="engine displacement",
    main="engine displacement distribution")
boxplot(mtcars$wt, xlab="weight", main="weight distribution")
boxplot(mtcars$hp, xlab="horsepower", main="horsepower distribution")
dev.off()

paste("Generando gráficas de dispersión para MTCARS")
png(file="mtcarsDispersion.png")
pairs(~disp+hp+wt, data=mtcars, main="MTCARS - Dispersion plots")
dev.off()
